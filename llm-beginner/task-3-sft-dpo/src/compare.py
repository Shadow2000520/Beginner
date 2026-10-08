import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint import load_adapter


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='同一指令对比 base / SFT / 可选 DPO')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/Qwen2.5-0.5B')
    parser.add_argument('--sft', type=Path, default=ROOT / 'ckpt/sft')
    parser.add_argument('--dpo', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--prompt', default='请用三句话向初学者解释什么是过拟合。')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.prompt or args.max_new_tokens <= 0:
        parser.error('prompt 不能为空，max-new-tokens 必须为正数')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA 不可用')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = tokenizer.apply_chat_template([dict(role='user', content=args.prompt)], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to(args.device)
    results = []
    variants = [('base', None), ('sft', args.sft)]
    if args.dpo:
        variants.append(('dpo', args.dpo))
    for name, adapter in variants:
        model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32)
        if adapter:
            load_adapter(model, adapter)
        model = model.to(args.device).eval()
        with torch.no_grad():
            ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')],
                                 pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(ids[0, inputs.input_ids.size(1):], skip_special_tokens=True)
        results.append(dict(model=name, adapter=str(adapter) if adapter else None, answer=answer))
        print(f'\n[{name}]\n{answer}', flush=True)
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(prompt=args.prompt, results=results), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
