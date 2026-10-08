import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.skill_loader import SkillLoader, read_metadata


class SkillTests(unittest.TestCase):
    def test_metadata_stops_before_body(self):
        stream = io.StringIO('---\nname: demo\ndescription: 测试\n---\n正文')
        self.assertEqual(read_metadata(stream)['name'], 'demo')
        self.assertEqual(stream.read(), '正文')

    def test_match_without_loading(self):
        loader = SkillLoader(ROOT / 'src/skills')
        with patch.object(loader, 'load', side_effect=AssertionError('不应加载正文')):
            self.assertEqual(loader.match('运行 pytest 修复测试失败', 1)[0]['name'], 'test-runner')
            self.assertEqual(loader.match('编写 PR 描述', 1)[0]['name'], 'pr-description-writer')
            self.assertEqual(loader.match('code review', 1)[0]['name'], 'code-review')
            self.assertEqual(loader.match('天气预报'), [])
        listing = loader.list_skills()
        listing[0]['name'] = 'changed'
        self.assertNotEqual(loader.list_skills()[0]['name'], 'changed')
        self.assertIn('run_tests', loader.load('test-runner'))
        with self.assertRaises(KeyError):
            loader.load('../README.md')

    def test_lazy_body_and_duplicate(self):
        with tempfile.TemporaryDirectory(prefix='skill-check-', dir=ROOT / 'data') as folder:
            root = Path(folder)
            path = root / 'demo/SKILL.md'
            path.parent.mkdir()
            header = '---\nname: demo\ndescription: test failure\n---\n'
            path.write_text(header + 'old', encoding='utf-8')
            loader = SkillLoader(root)
            path.write_text(header + 'new', encoding='utf-8')
            self.assertEqual(loader.load('demo'), 'new')
            other = root / 'another/SKILL.md'
            other.parent.mkdir()
            other.write_text(header + 'body', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '重复'):
                SkillLoader(root)

    def test_bad_frontmatter(self):
        for text in ['name: demo', '---\nname: demo\n---\nbody',
                     '---\nname: demo\ndescription: 123\n---',
                     '---\nname: ../bad\ndescription: test\n---',
                     '---\nname: demo\ndescription: test']:
            with self.assertRaises(ValueError):
                read_metadata(io.StringIO(text))


if __name__ == '__main__':
    unittest.main()
