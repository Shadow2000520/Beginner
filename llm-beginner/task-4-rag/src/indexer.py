import argparse
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

import faiss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.chunker import chunk_pdf
from src.embedding import Embedder


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_index(pdf_path, model_path, output, chunk_size=384, overlap=64, batch_size=32, device='cpu'):
    pdf_path, model_path, output = Path(pdf_path).resolve(), Path(model_path).resolve(), Path(output).resolve()
    print('从 PDF 提取文本并切分……', flush=True)
    chunks = chunk_pdf(pdf_path, chunk_size, overlap)
    print(f'生成 {len(chunks)} 个片段，正在计算 embedding……', flush=True)
    embedder = Embedder(model_path, device)
    vectors = embedder.encode([c['text'] for c in chunks], batch_size)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    output.mkdir(parents=True, exist_ok=True)
    # 普通目录继承输出目录权限，避免 Windows 临时目录限制其他账户读取。
    staging = output / f'build-{uuid4().hex}'
    staging.mkdir()
    try:
        faiss.write_index(index, str(staging / 'index.faiss'))
        (staging / 'chunks.json').write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
        metadata = dict(version=1, pdf=str(pdf_path), pdf_sha256=file_hash(pdf_path),
                        embedding_model=str(model_path), normalized=True, metric='inner_product',
                        chunk_size=chunk_size, overlap=overlap, count=len(chunks), dimension=vectors.shape[1],
                        files={name: file_hash(staging / name) for name in ['index.faiss', 'chunks.json']})
        (staging / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        for name in ['index.faiss', 'chunks.json', 'metadata.json']:
            (staging / name).replace(output / name)
    finally:
        for name in ['index.faiss', 'chunks.json', 'metadata.json']:
            (staging / name).unlink(missing_ok=True)
        staging.rmdir()
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从 PDF 建立 BGE + FAISS 索引')
    parser.add_argument('--pdf', type=Path, default=ROOT / 'data/kb.pdf')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/bge-small-zh-v1.5')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/index')
    parser.add_argument('--chunk-size', type=int, default=384)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('batch-size 必须是正整数')
    result = build_index(args.pdf, args.model, args.output, args.chunk_size, args.overlap, args.batch_size, args.device)
    print(f'索引已保存：{args.output.resolve()}；片段数={result["count"]}，维度={result["dimension"]}')
