import fnmatch
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL_SCHEMA = {'type': 'function', 'function': {
    'name': 'file_search', 'description': '在任务五目录内按文件名通配符或内容关键词搜索，返回路径、匹配总数和内容片段。dir 可为目录或文件；读取文件用 pattern="*"。',
    'parameters': {'type': 'object', 'properties': {'pattern': {'type': 'string'}, 'dir': {'type': 'string'}},
                   'required': ['pattern', 'dir'], 'additionalProperties': False}}}


def run(args):
    pattern, directory = args.get('pattern'), args.get('dir')
    if not isinstance(pattern, str) or not pattern.strip() or not isinstance(directory, str) or not directory.strip():
        raise ValueError('pattern 和 dir 必须是非空字符串')
    target = (ROOT / directory).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('路径超出任务五目录')
    if not target.exists():
        raise FileNotFoundError(directory)
    matches, count, scanned = [], 0, 0
    paths = [target] if target.is_file() else target.rglob('*')
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            continue
        scanned += 1
        if scanned > 10000:
            raise ValueError('文件过多，请缩小 dir 范围')
        relative = path.relative_to(ROOT).as_posix()
        name_hit = fnmatch.fnmatch(path.name.lower(), pattern.lower()) or pattern.lower() in path.name.lower()
        content = ''
        if path.stat().st_size <= 1024 * 1024:
            try:
                content = path.read_text(encoding='utf-8-sig')
            except (UnicodeError, OSError):
                pass
        position = content.lower().find(pattern.lower())
        if name_hit or position >= 0:
            count += 1
            if len(matches) < 50:
                start = 0 if name_hit else max(0, position - 100)
                matches.append(dict(path=relative, snippet=content[start:start + 1000],
                                    content_available=bool(content)))
    return json.dumps(dict(count=count, matches=matches, truncated=count > len(matches)), ensure_ascii=False)
