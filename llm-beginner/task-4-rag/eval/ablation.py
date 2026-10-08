import argparse
import gc
import json
from pathlib import Path
import runpy
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CHECKS = runpy.run_path(str(ROOT / 'eval/run.py'))

import torch
from src.generator import Generator, prepare_context
from src.indexer import build_index, file_hash
from src.retriever import Retriever
from _eval_harness import setup_console


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


class RecordedRetriever:
    def __init__(self, retriever):
        self.retriever = retriever
        self.results = {}

    def retrieve(self, query, k=10):
        results = self.retriever.retrieve(query, k)
        self.results[query] = results
        return results


def retrieval_experiment(args):
    gold, error = CHECKS['load_gold_qa']()
    if error or not gold:
        raise ValueError(error or 'gold QA 为空')
    report = dict(pdf_sha256=file_hash(ROOT / 'data/kb.pdf'),
                  gold_sha256=file_hash(ROOT / 'data/gold_qa.jsonl'),
                  overlap=64, candidate_k=40, device=args.device, runs=[])
    for size in [128, 256, 384]:
        with tempfile.TemporaryDirectory(prefix='ablation-', dir=ROOT / 'data') as folder:
            metadata = build_index(ROOT / 'data/kb.pdf', ROOT / 'models/bge-small-zh-v1.5',
                                   folder, size, 64, 32, args.device)
            retriever = Retriever(folder, device=args.device, candidate_k=40,
                                  reranker_batch_size=args.batch_size)
            reranker = retriever.reranker
            for enabled in [False, True]:
                retriever.reranker = reranker if enabled else None
                recorded = RecordedRetriever(retriever)
                started = time.perf_counter()
                result = CHECKS['test_nndl_gold_recall_at_10'](recorded)
                result.update(chunk_size=size, rerank=enabled, chunks=metadata['count'],
                              elapsed_seconds=round(time.perf_counter() - started, 2))
                result['queries'] = [dict(id=item['id'], question=item['question'],
                                          results=recorded.results[item['question']]) for item in gold]
                report['runs'].append(result)
                summary = {k: v for k, v in result.items() if k not in ['details', 'queries']}
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            del recorded, retriever, reranker
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save(ROOT / 'eval/ablation_retrieval.json', report)


def generation_experiment(args):
    report = json.loads((ROOT / 'eval/ablation_retrieval.json').read_text(encoding='utf-8'))
    if report['pdf_sha256'] != file_hash(ROOT / 'data/kb.pdf') or report['gold_sha256'] != file_hash(ROOT / 'data/gold_qa.jsonl'):
        raise ValueError('PDF 或 gold QA 已变化，请重新运行检索消融')
    runs = {item['rerank']: item for item in report['runs'] if item['chunk_size'] == 384}
    if set(runs) != {False, True}:
        raise ValueError('缺少 384 字符的有/无 rerank 对照')
    questions = runs[True]['queries']
    count = min(args.generation_samples, len(questions))
    selected = [questions[i * len(questions) // count]['id'] for i in range(count)]
    generator = Generator()
    output = dict(model=generator.model, base_url=generator.base_url,
                  retrieval_sha256=file_hash(ROOT / 'eval/ablation_retrieval.json'),
                  chunk_size=384, top_k=5, max_context_chars=4000,
                  temperature=0, max_tokens=512, question_ids=selected, runs=[])
    variants = [('no_rerank_grounded', False, True), ('rerank_grounded', True, True),
                ('rerank_basic_prompt', True, False)]
    for name, rerank, grounded in variants:
        queries = {item['id']: item for item in runs[rerank]['queries']}
        for ident in selected:
            item = queries[ident]
            context, sources = prepare_context(item['results'][:5], 4000)
            started = time.perf_counter()
            response = generator.generate(item['question'], context, grounded=grounded)
            output['runs'].append(dict(variant=name, id=ident, question=item['question'],
                                       answer=response, sources=sources, context=context,
                                       elapsed_seconds=round(time.perf_counter() - started, 2),
                                       review=dict(total_claims=None, supported_claims=None,
                                                   answers_question=None, notes='')))
            print(f'[{name}] {ident}\n{response}\n', flush=True)
    save(ROOT / 'eval/ablation_generation.json', output)


if __name__ == '__main__':
    setup_console()
    parser = argparse.ArgumentParser(description='固定 gold QA 的 chunk / rerank / prompt 消融')
    parser.add_argument('--stage', choices=['retrieval', 'generation', 'all'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--generation-samples', type=int, default=30)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.generation_samples <= 0:
        parser.error('batch-size 和 generation-samples 必须大于 0')
    if args.stage in ['retrieval', 'all']:
        retrieval_experiment(args)
    if args.stage in ['generation', 'all']:
        generation_experiment(args)
    print('实验完成，结果位于 eval/ablation_retrieval.json 或 eval/ablation_generation.json')
