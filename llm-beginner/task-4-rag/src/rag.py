import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generator import Generator, prepare_context
from src.retriever import Retriever


@lru_cache(maxsize=1)
def default_retriever(device):
    return Retriever(device=device)


def answer(query, *, k=5, max_context_chars=4000, device=None, retriever=None, generator=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('query 不能为空')
    if type(k) is not int or k <= 0:
        raise ValueError('k 必须是正整数')
    if type(max_context_chars) is not int or max_context_chars <= 0:
        raise ValueError('max_context_chars 必须是正整数')
    if retriever is None:
        retriever = default_retriever(device or os.getenv('RAG_DEVICE', 'cpu'))
    context, sources = prepare_context(retriever.retrieve(query, k=k), max_context_chars)
    if not sources:
        return {'answer': '根据现有资料无法确定。', 'sources': []}
    generator = generator if generator is not None else Generator()
    return {'answer': generator.generate(query, context), 'sources': sources}


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='PDF 教材问答')
    parser.add_argument('--query', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=os.getenv('RAG_DEVICE', 'cpu'))
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max-context-chars', type=int, default=4000)
    parser.add_argument('--base-url')
    parser.add_argument('--model')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = answer(args.query, k=args.k, max_context_chars=args.max_context_chars,
                    device=args.device, generator=Generator(args.base_url, args.model))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
