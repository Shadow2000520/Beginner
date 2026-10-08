import json
import re

import requests

TOOL_SCHEMA = {'type': 'function', 'function': {
    'name': 'wiki', 'description': '查询中英文维基百科条目简介，返回标题、URL 和摘要；中文问题优先中文站点。',
    'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                   'required': ['query'], 'additionalProperties': False}}}


def request(language, **params):
    response = requests.get(f'https://{language}.wikipedia.org/w/api.php',
                            params=dict(action='query', format='json', **params),
                            headers={'User-Agent': 'LLMBeginnerTeaching/1.0'}, timeout=15)
    response.raise_for_status()
    result = response.json()
    if 'error' in result:
        raise RuntimeError('Wikipedia API: ' + str(result['error']))
    return result


def page(language, title):
    result = request(language, titles=title, redirects=1, prop='extracts|info',
                     exintro=1, explaintext=1, inprop='url')
    for item in result.get('query', {}).get('pages', {}).values():
        if 'missing' not in item and item.get('extract', '').strip():
            return dict(title=item['title'], url=item['fullurl'], language=language,
                        summary=item['extract'][:6000])
    return None


def run(args):
    query = args.get('query')
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        raise ValueError('query 必须是 1–200 字符的条目名称')
    language = 'zh' if re.search(r'[\u4e00-\u9fff]', query) else 'en'
    found = page(language, query.strip())
    if found is None:
        result = request(language, list='search', srsearch=query, srlimit=3)
        for item in result.get('query', {}).get('search', []):
            found = page(language, item['title'])
            if found:
                break
    if found is None:
        raise LookupError('维基百科没有找到匹配条目，请换用更准确的条目名称')
    return json.dumps(found, ensure_ascii=False)
