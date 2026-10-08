"""Train and evaluate prompt-based entity and relation extraction on DuIE examples."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from model import UIE

ROOT = Path(__file__).resolve().parent
THRESHOLDS = [.2, .3, .4, .5, .6, .7, .8]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def prepare(output, seed):
    source = ROOT/'data/DuIE'
    original_train, original_dev = read_rows(source/'train.txt'), read_rows(source/'dev.txt')
    if (source/'test.txt').stat().st_size:
        raise ValueError('The repository test file is no longer empty; recheck the split')
    contents = sorted({r['content'] for r in original_train})
    train_contents, dev_contents = train_test_split(contents, test_size=10, random_state=seed)
    train_set, dev_set = set(train_contents), set(dev_contents)
    test_set = {r['content'] for r in original_dev}
    assert not (train_set & dev_set or train_set & test_set or dev_set & test_set)
    splits = {'train': [r for r in original_train if r['content'] in train_set],
              'dev': [r for r in original_train if r['content'] in dev_set],
              'test': original_dev}
    for split, rows in splits.items():
        path = output/'data'/f'{split}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
    for rows in splits.values():
        for row in rows:
            for answer in row['result_list']:
                if row['content'][answer['start']:answer['end']] != answer['text']:
                    raise ValueError(f'Invalid gold span: {answer}')
    audit = {'seed': seed, 'original': {'train_rows': len(original_train), 'dev_rows': len(original_dev), 'test_rows': 0},
             'splits': {name: {'rows': len(rows), 'contents': len({r['content'] for r in rows}),
                                'positive_rows': sum(bool(r['result_list']) for r in rows),
                                'gold_spans': sum(len(r['result_list']) for r in rows),
                                'relation_rows': sum('的' in r['prompt'] for r in rows)} for name, rows in splits.items()},
             'source_sha256': {f'{name}.txt': hashlib.sha256((source/f'{name}.txt').read_bytes()).hexdigest() for name in ('train','dev','test')}}
    write_json(output/'data_audit.json', audit)
    return splits, audit


def encode_row(tokenizer, row, max_length):
    encoded = tokenizer(row['prompt'], row['content'], truncation='only_second', max_length=max_length,
                        padding='max_length', return_offsets_mapping=True)
    if not hasattr(encoded, 'sequence_ids'):
        raise TypeError('A fast tokenizer with sequence_ids() is required')
    sequence_ids = encoded.sequence_ids()
    offsets = encoded['offset_mapping']
    content_mask = [1 if side == 1 and end > start else 0 for side, (start, end) in zip(sequence_ids, offsets)]
    start_labels, end_labels = [0.] * len(offsets), [0.] * len(offsets)
    omitted, unaligned = 0, 0
    for answer in row['result_list']:
        start, end = answer['start'], answer['end']
        starts = [i for i, ((a, b), mask) in enumerate(zip(offsets, content_mask)) if mask and a <= start < b]
        ends = [i for i, ((a, b), mask) in enumerate(zip(offsets, content_mask)) if mask and a < end <= b]
        if not starts or not ends:
            omitted += 1
            continue
        s, e = starts[0], ends[-1]
        if s > e:
            omitted += 1
            continue
        start_labels[s] = 1.
        end_labels[e] = 1.
        if offsets[s][0] != start or offsets[e][1] != end:
            unaligned += 1
    return {'input_ids': torch.tensor(encoded['input_ids']),
            'token_type_ids': torch.tensor(encoded.get('token_type_ids', [0]*len(offsets))),
            'attention_mask': torch.tensor(encoded['attention_mask']),
            'content_mask': torch.tensor(content_mask, dtype=torch.float32),
            'start_labels': torch.tensor(start_labels), 'end_labels': torch.tensor(end_labels),
            'offsets': offsets, 'row': row, 'omitted': omitted, 'unaligned': unaligned}


def encode_rows(tokenizer, rows, max_length):
    return [encode_row(tokenizer, r, max_length) for r in rows]


def collate(rows):
    keys = ('input_ids','token_type_ids','attention_mask','content_mask','start_labels','end_labels')
    return {**{k: torch.stack([r[k] for r in rows]) for k in keys},
            'offsets': [r['offsets'] for r in rows], 'rows': [r['row'] for r in rows]}


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def autocast(device, enabled):
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled and device.type == 'cuda' else nullcontext()


def spans_from_probs(start_probs, end_probs, offsets, mask, threshold):
    starts = [i for i, (p, valid) in enumerate(zip(start_probs, mask)) if valid and p >= threshold]
    ends = [i for i, (p, valid) in enumerate(zip(end_probs, mask)) if valid and p >= threshold]
    pairs = []
    for end in ends:
        available = [s for s in starts if s <= end]
        if available:
            start = available[-1]
            starts.remove(start)
            pairs.append((offsets[start][0], offsets[end][1]))
    return sorted(set(pairs))


def score_records(records, threshold):
    counts = {'all': [0, 0, 0], 'entity': [0, 0, 0], 'relation': [0, 0, 0]}
    predictions = []
    for rec in records:
        predicted = set(spans_from_probs(rec['start_probs'], rec['end_probs'], rec['offsets'], rec['mask'], threshold))
        gold = {(x['start'], x['end']) for x in rec['row']['result_list']}
        category = 'relation' if '的' in rec['row']['prompt'] else 'entity'
        for key in ('all', category):
            counts[key][0] += len(predicted & gold)
            counts[key][1] += len(predicted)
            counts[key][2] += len(gold)
        predictions.append({'content': rec['row']['content'], 'prompt': rec['row']['prompt'],
                            'gold': [{'text': rec['row']['content'][s:e], 'start': s, 'end': e} for s,e in sorted(gold)],
                            'predicted': [{'text': rec['row']['content'][s:e], 'start': s, 'end': e} for s,e in sorted(predicted)]})
    metrics = {}
    for key, (correct, inferred, labeled) in counts.items():
        precision = correct/inferred if inferred else 0.
        recall = correct/labeled if labeled else 0.
        metrics[key] = {'precision': precision, 'recall': recall, 'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
                        'correct': correct, 'predicted': inferred, 'gold': labeled}
    return metrics, predictions


@torch.inference_mode()
def collect_records(model, dataset, device, batch_size, amp):
    model.eval()
    records = []
    for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collate):
        inputs = to_device(batch, device)
        with autocast(device, amp):
            start, end = model(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
        starts, ends = start.float().cpu().tolist(), end.float().cpu().tolist()
        for row, offsets, mask, sp, ep in zip(batch['rows'], batch['offsets'], batch['content_mask'].tolist(), starts, ends):
            records.append({'row': row, 'offsets': offsets, 'mask': mask, 'start_probs': sp, 'end_probs': ep})
    return records


def select_threshold(records):
    scored = [(t, score_records(records, t)[0]['all']['f1']) for t in THRESHOLDS]
    return max(scored, key=lambda v: (v[1], -abs(v[0]-.5)))[0]


def save_checkpoint(path, model, tokenizer, metadata):
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    save_file({k: v.detach().cpu().contiguous() for k,v in model.state_dict().items()}, str(path/'uie.safetensors'))
    write_json(path/'uie_config.json', metadata)


def load_checkpoint(path, device):
    info = json.loads((path/'uie_config.json').read_text(encoding='utf-8'))
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(path, local_files_only=True))
    model = UIE(encoder)
    model.load_state_dict(load_file(str(path/'uie.safetensors')))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer, info


def swap_spo(rows, seed):
    """Safe subset of the repository's rule: swap only unambiguous subject/object strings."""
    rng = random.Random(seed)
    by_predicate = defaultdict(list)
    for row in rows:
        if '的' in row['prompt'] and len(row['result_list']) == 1:
            by_predicate[row['prompt'].split('的')[-1]].append(row)
    synthetic = []
    existing = {(r['content'],r['prompt']) for r in rows}
    for predicate, examples in by_predicate.items():
        if len(examples) < 2:
            continue
        for source in examples:
            subject, answer = source['prompt'].rsplit('的', 1)[0], source['result_list'][0]['text']
            candidates = [r for r in examples if r is not source]
            candidate = rng.choice(candidates)
            old_subject, old_object = candidate['prompt'].rsplit('的',1)[0], candidate['result_list'][0]['text']
            if not all([subject, answer, old_subject, old_object]) or old_subject == old_object:
                continue
            text = candidate['content']
            if text.count(old_subject) != 1 or text.count(old_object) != 1 or answer in text:
                continue
            text = text.replace(old_subject, '\uFFF0').replace(old_object, '\uFFF1').replace('\uFFF0', subject).replace('\uFFF1', answer)
            if text.count(answer) != 1 or (text, source['prompt']) in existing:
                continue
            start = text.index(answer)
            item = {'content': text, 'prompt': source['prompt'], 'result_list': [{'text': answer, 'start': start, 'end': start+len(answer)}]}
            synthetic.append(item)
            existing.add((text, source['prompt']))
    return synthetic


def hard_negatives(rows, model, tokenizer, max_length, device, amp, count=150):
    negatives = [r for r in rows if '的' in r['prompt'] and not r['result_list']]
    encoded = encode_rows(tokenizer, negatives, max_length)
    records = collect_records(model, encoded, device, 32, amp)
    ranked = []
    for row, rec in zip(negatives, records):
        starts = [p for p,m in zip(rec['start_probs'],rec['mask']) if m]
        ends = [p for p,m in zip(rec['end_probs'],rec['mask']) if m]
        ranked.append((max(starts, default=0.)*max(ends, default=0.), row))
    ranked.sort(key=lambda x:-x[0])
    return [r for _,r in ranked[:min(count,len(ranked))]]


def train_variant(name, rows, dev_encoded, tokenizer, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    encoder = AutoModel.from_pretrained(args.model, local_files_only=args.local_only)
    model = UIE(encoder).to(device)
    train_encoded = encode_rows(tokenizer, rows, args.max_length)
    path = args.output/name
    path.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(train_encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    steps = len(loader)*args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.06*steps), steps)
    config = {'variant': name, 'model': str(Path(args.model).resolve()) if Path(args.model).exists() else args.model,
              'seed': args.seed, 'train_rows': len(rows), 'epochs': args.epochs, 'batch_size': args.batch_size,
              'max_length': args.max_length, 'learning_rate': args.lr, 'amp': args.amp, 'device': str(device),
              'omitted_gold_spans': sum(r['omitted'] for r in train_encoded), 'unaligned_gold_spans': sum(r['unaligned'] for r in train_encoded)}
    write_json(path/'config.json', config)
    history, best = [], -1.
    begin = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        losses = []
        for batch in loader:
            batch = to_device(batch,device)
            with autocast(device,args.amp):
                start,end = model(batch['input_ids'],batch['token_type_ids'],batch['attention_mask'],return_logits=True)
                mask = batch['content_mask']
                start_loss = torch.nn.functional.binary_cross_entropy_with_logits(start.float(),batch['start_labels'],reduction='none',pos_weight=torch.tensor(8.,device=device))
                end_loss = torch.nn.functional.binary_cross_entropy_with_logits(end.float(),batch['end_labels'],reduction='none',pos_weight=torch.tensor(8.,device=device))
                loss = ((start_loss+end_loss)*mask).sum()/(2*mask.sum())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite UIE loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        records = collect_records(model,dev_encoded,device,args.eval_batch_size,args.amp)
        threshold = select_threshold(records)
        dev_metrics,_ = score_records(records,threshold)
        entry = {'epoch':epoch,'loss':float(np.mean(losses)),'threshold':threshold,'dev':dev_metrics}
        history.append(entry)
        print(f'{name} epoch={epoch}/{args.epochs} loss={entry["loss"]:.4f} dev_f1={dev_metrics["all"]["f1"]:.4f} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
        if dev_metrics['all']['f1'] > best:
            best = dev_metrics['all']['f1']
            save_checkpoint(path/'best',model,tokenizer,{**config,'threshold':threshold,'best_epoch':epoch})
        write_json(path/'history.json',history)
    return {'variant':name,'best_dev_f1':best,'train_rows':len(rows),'best_epoch':json.loads((path/'best/uie_config.json').read_text(encoding='utf-8'))['best_epoch']}


@torch.inference_mode()
def predict(model,tokenizer,content,prompt,max_length,threshold,device):
    row={'content':content,'prompt':prompt,'result_list':[]}
    record=collect_records(model,[encode_row(tokenizer,row,max_length)],device,1,False)[0]
    spans=spans_from_probs(record['start_probs'],record['end_probs'],record['offsets'],record['mask'],threshold)
    return [{'text':content[s:e],'start':s,'end':e} for s,e in spans]


def predict_schema(model,tokenizer,content,schema,max_length,threshold,device):
    """A list extracts entity types; a dict extracts subjects then their attributes."""
    if isinstance(schema,list) and all(isinstance(label,str) for label in schema):
        return {label:predict(model,tokenizer,content,label,max_length,threshold,device) for label in schema}
    if isinstance(schema,dict) and all(isinstance(key,str) and isinstance(values,list) and all(isinstance(v,str) for v in values)
                                       for key,values in schema.items()):
        output={}
        for subject_type,relations in schema.items():
            for subject in predict(model,tokenizer,content,subject_type,max_length,threshold,device):
                output[subject['text']]={relation:[answer for answer in predict(model,tokenizer,content,f"{subject['text']}的{relation}",max_length,threshold,device)
                                                   if (answer['start'],answer['end']) != (subject['start'],subject['end'])]
                                         for relation in relations}
        return output
    raise ValueError('schema must be a JSON list of entity types or a mapping from subject type to relation names')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','predict'])
    parser.add_argument('--model',default='BAAI/bge-small-zh-v1.5')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/full')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--content')
    parser.add_argument('--prompt')
    parser.add_argument('--schema',help='JSON list of entity types or JSON mapping of subjects to relation names')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--local-only',action='store_true')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--eval-batch-size',type=int,default=32)
    parser.add_argument('--max-length',type=int,default=256)
    parser.add_argument('--lr',type=float,default=5e-5)
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    if args.command=='predict':
        if not args.checkpoint or not args.content or not (args.prompt or args.schema):
            parser.error('predict needs --checkpoint, --content, and --prompt or --schema')
        model,tokenizer,info=load_checkpoint(args.checkpoint,device)
        if args.schema:
            schema=json.loads(args.schema)
            result={'content':args.content,'schema':schema,'results':predict_schema(model,tokenizer,args.content,schema,info['max_length'],info['threshold'],device)}
        else:
            result={'content':args.content,'prompt':args.prompt,'answers':predict(model,tokenizer,args.content,args.prompt,info['max_length'],info['threshold'],device)}
        print(json.dumps(result,ensure_ascii=False))
        return
    splits,audit=prepare(args.output,args.seed)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    if args.command=='prepare': return
    if (args.output/'result.json').exists():
        raise FileExistsError('Completed run already exists; choose a new --output')
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=args.local_only,use_fast=True)
    dev_encoded=encode_rows(tokenizer,splits['dev'],args.max_length)
    test_encoded=encode_rows(tokenizer,splits['test'],args.max_length)
    alignment={'dev':{'omitted':sum(r['omitted'] for r in dev_encoded),'unaligned':sum(r['unaligned'] for r in dev_encoded)},
               'test':{'omitted':sum(r['omitted'] for r in test_encoded),'unaligned':sum(r['unaligned'] for r in test_encoded)}}
    if alignment['dev']['omitted'] or alignment['test']['omitted']:
        raise ValueError(f'Gold spans were truncated: {alignment}')
    write_json(args.output/'alignment_audit.json',alignment)
    experiments=[]
    experiments.append(train_variant('baseline',splits['train'],dev_encoded,tokenizer,args))
    swap=swap_spo(splits['train'],args.seed)
    print(f'swap_spo generated {len(swap)} validated rows',flush=True)
    write_json(args.output/'swap_samples.json',swap)
    if swap:
        experiments.append(train_variant('swap_spo',splits['train']+swap,dev_encoded,tokenizer,args))
    base_model,_,_=load_checkpoint(args.output/'baseline/best',device)
    mined=hard_negatives(splits['train'],base_model,tokenizer,args.max_length,device,args.amp)
    print(f'hard-negative mining selected {len(mined)} labeled training negatives',flush=True)
    write_json(args.output/'hard_negatives.json',mined)
    if mined:
        experiments.append(train_variant('hard_negatives',splits['train']+mined*2,dev_encoded,tokenizer,args))
    chosen=max(experiments,key=lambda x:x['best_dev_f1'])
    model,tokenizer,info=load_checkpoint(args.output/chosen['variant']/'best',device)
    dev_records=collect_records(model,dev_encoded,device,args.eval_batch_size,args.amp)
    test_records=collect_records(model,test_encoded,device,args.eval_batch_size,args.amp)
    dev_metrics,_=score_records(dev_records,info['threshold'])
    test_metrics,predictions=score_records(test_records,info['threshold'])
    write_json(args.output/'test_predictions.json',predictions)
    demos=[]
    for row in [r for r in splits['test'] if r['result_list']][:4]:
        demos.append({'content':row['content'],'prompt':row['prompt'],'gold':row['result_list'],
                      'predicted':predict(model,tokenizer,row['content'],row['prompt'],info['max_length'],info['threshold'],device)})
    example='谭孝曾是谭元寿的长子，也是谭派第六代传人。'
    demos.append({'content':example,'entity_prompt':'人物','entity_prediction':predict(model,tokenizer,example,'人物',info['max_length'],info['threshold'],device),
                  'relation_prompt':'谭孝曾的父亲','relation_prediction':predict(model,tokenizer,example,'谭孝曾的父亲',info['max_length'],info['threshold'],device),
                  'ner_schema_result':predict_schema(model,tokenizer,example,['人物'],info['max_length'],info['threshold'],device),
                  'relation_schema_result':predict_schema(model,tokenizer,example,{'人物':['父亲']},info['max_length'],info['threshold'],device)})
    write_json(args.output/'demo.json',demos)
    result={'selected_variant':chosen['variant'],'variants':experiments,'threshold':info['threshold'],'dev':dev_metrics,'test':test_metrics,
            'alignment':alignment,'swap_samples':len(swap),'hard_negatives':len(mined),'test_rows':len(splits['test'])}
    write_json(args.output/'result.json',result)
    print('COMPLETE '+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
