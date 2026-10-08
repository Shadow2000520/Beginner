from pathlib import Path
import re

import yaml


def read_metadata(stream):
    if stream.readline().strip() != '---':
        raise ValueError('SKILL.md 必须以 YAML front-matter 开头')
    lines = []
    for _ in range(100):
        line = stream.readline()
        if line.strip() == '---':
            metadata = yaml.safe_load(''.join(lines))
            if not isinstance(metadata, dict):
                break
            if not all(isinstance(metadata.get(k), str) and metadata[k].strip()
                       for k in ('name', 'description')):
                break
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', metadata['name']):
                raise ValueError('Skill name 只能使用小写字母、数字和连字符')
            return {k: metadata[k].strip() for k in ('name', 'description')}
        if not line or sum(map(len, lines)) + len(line) > 8192:
            break
        lines.append(line)
    raise ValueError('无效的 front-matter：需要 name、description 和结束分隔符')


def terms(text):
    tokens = set(re.findall(r'[a-z][a-z0-9_-]*', text.lower()))
    for phrase in re.findall(r'[\u4e00-\u9fff]+', text):
        tokens.update(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return tokens


class SkillLoader:
    def __init__(self, skills_dir):
        self.root = Path(skills_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f'Skill 目录不存在：{self.root}')
        self.skills = {}
        for path in sorted(self.root.glob('*/SKILL.md')):
            self.check_path(path)
            with path.open(encoding='utf-8-sig') as stream:
                metadata = read_metadata(stream)
            name = metadata['name']
            if name in self.skills:
                raise ValueError(f'Skill name 重复：{name}')
            self.skills[name] = (metadata, path)

    def check_path(self, path):
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Skill 路径超出 skills 目录')

    def list_skills(self):
        return [dict(metadata) for metadata, _ in self.skills.values()]

    def match(self, task, limit=2):
        if not isinstance(task, str) or type(limit) is not int or limit <= 0:
            raise ValueError('task 必须为字符串，limit 必须为正整数')
        query = terms(task)
        scored = [(len(query & terms(item['description'])), item) for item in self.list_skills()]
        scored.sort(key=lambda pair: (-pair[0], pair[1]['name']))
        return [item for score, item in scored if score > 0][:limit]

    def load(self, name):
        if name not in self.skills:
            raise KeyError(f'未知 Skill：{name}')
        metadata, path = self.skills[name]
        self.check_path(path)
        with path.open(encoding='utf-8-sig') as stream:
            if read_metadata(stream) != metadata:
                raise ValueError('Skill 元数据已变化，请重新创建 SkillLoader')
            body = stream.read().strip()
        if not body:
            raise ValueError(f'Skill 正文为空：{name}')
        return body
