import re
from pathlib import Path
from contextlib import nullcontext

import pypdf
from pypdf import PdfReader


def chunk_spans(text, chunk_size=384, overlap=64):
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size 必须是正整数')
    if type(overlap) is not int or not 0 <= overlap < chunk_size:
        raise ValueError('overlap 必须满足 0 <= overlap < chunk_size')
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            lower = start + max(overlap + 1, int(chunk_size * .7))
            boundaries = [m.end() + lower for m in re.finditer(r'[。！？；\n]', text[lower:end])]
            if boundaries:
                end = boundaries[-1]
        if text[start:end].strip():
            yield start, end
        if end == len(text):
            break
        start = end - overlap


def chunk_text(text, chunk_size=384, overlap=64):
    return [text[start:end] for start, end in chunk_spans(text, chunk_size, overlap)]


def extract_pdf_pages(path):
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, 1):
        # 教材中的复杂矢量图会触发新版 pypdf 默认的 5000 次上限。
        config = getattr(pypdf, 'apply_configuration', None)
        context = config(xform_maximum_invocations_per_extraction=20000) if config else nullcontext()
        with context:
            text = page.extract_text() or ''
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')
        if text.strip():
            pages.append(dict(page=number, text=text.strip()))
    if not pages:
        raise ValueError('PDF 没有可提取文本，扫描件需要先做 OCR')
    return pages


def chunk_pdf(path, chunk_size=384, overlap=64):
    from bisect import bisect_right

    path = Path(path)
    pages = extract_pdf_pages(path)
    starts, sections, offset = [], [], 0
    for page in pages:
        starts.append(offset)
        sections.append(page['text'] + '\n')
        offset += len(sections[-1])
    text = ''.join(sections)
    chunks = []
    for start, end in chunk_spans(text, chunk_size, overlap):
        first = pages[bisect_right(starts, start) - 1]['page']
        last = pages[bisect_right(starts, end - 1) - 1]['page']
        chunks.append(dict(id=len(chunks), text=text[start:end], page_start=first, page_end=last,
                           char_start=start, char_end=end, source=f'{path.name}:PDF页{first}-{last}'))
    return chunks
