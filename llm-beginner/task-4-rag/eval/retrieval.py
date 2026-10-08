import argparse
import json
from pathlib import Path
import runpy


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    checks = runpy.run_path(str(root / 'eval/run.py'))
    from src.retriever import Retriever
    from _eval_harness import setup_console

    setup_console()
    parser = argparse.ArgumentParser(description='M3：固定 gold QA 检索评测')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--candidate-k', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--no-rerank', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()
    retriever = Retriever(device=args.device, use_reranker=not args.no_rerank,
                          candidate_k=args.candidate_k, reranker_batch_size=args.batch_size)
    result = checks['test_nndl_gold_recall_at_10'](retriever)
    result['config'] = dict(rerank=not args.no_rerank, candidate_k=args.candidate_k,
                          batch_size=args.batch_size, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.no_save:
        output = root / 'eval/retrieval_result.json'
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'结果写入 {output}')
    raise SystemExit(0 if result['pass'] else 1)
