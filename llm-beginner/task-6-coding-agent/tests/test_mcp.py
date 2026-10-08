import difflib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mcp_server import list_tools


def unpack(result):
    if result.model_dump(by_alias=True)["isError"]:
        raise AssertionError(str(result.content))
    return json.loads(next(x.text for x in result.content if x.type == 'text'))


class MCPTests(unittest.IsolatedAsyncioTestCase):
    def test_schema(self):
        tools = list_tools()
        self.assertEqual({x['name'] for x in tools}, {'read_file', 'write_file', 'run_tests', 'git_diff', 'git_apply'})
        self.assertTrue(all(x['description'] and x['input_schema']['type'] == 'object' for x in tools))

    async def test_stdio_tools(self):
        with tempfile.TemporaryDirectory(prefix='mcp-check-', dir=ROOT / 'data') as directory:
            repo = Path(directory)
            buggy = 'def add(a, b):\n    return a - b\n'
            fixed = buggy.replace('a - b', 'a + b')
            (repo / 'calculator.py').write_text(buggy, encoding='utf-8')
            (repo / 'test_calculator.py').write_text('from calculator import add\ndef test_add():\n    assert add(2, 3) == 5\n', encoding='utf-8')
            subprocess.run(['git', 'init', '-q'], cwd=repo, check=True, capture_output=True)
            subprocess.run(['git', 'add', '.'], cwd=repo, check=True, capture_output=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
            params = StdioServerParameters(command=sys.executable,
                      args=['-B', str(ROOT / 'src/mcp_server.py'), '--repo', str(repo)], env=env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=90) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    self.assertEqual(len(tools.tools), 5)
                    result = unpack(await session.call_tool('read_file', {'path': 'calculator.py'}))
                    self.assertEqual(result['content'], buggy)
                    result = unpack(await session.call_tool('run_tests', {}))
                    self.assertFalse(result['tests_passed'])
                    for path in ['../README.md', str(ROOT / 'README.md'), '.git/config']:
                        self.assertTrue((await session.call_tool('read_file', {'path': path})).model_dump(by_alias=True)["isError"])
                    for path in ['test_calculator.py', 'calculator.py.orig', '../escape.py', 'conftest.py']:
                        self.assertTrue((await session.call_tool('write_file', {'path': path, 'content': ''})).model_dump(by_alias=True)["isError"])
                    result = unpack(await session.call_tool('write_file', {'path': 'calculator.py', 'content': fixed}))
                    self.assertTrue(result['written'])
                    self.assertTrue(unpack(await session.call_tool('run_tests', {}))['tests_passed'])
                    self.assertIn('+    return a + b', unpack(await session.call_tool('git_diff', {}))['patch'])
                    patch = ''.join(difflib.unified_diff(buggy.splitlines(True), fixed.splitlines(True),
                                                       fromfile='a/calculator.py', tofile='b/calculator.py'))
                    self.assertTrue((await session.call_tool('git_apply', {'patch': patch})).model_dump(by_alias=True)["isError"])
                    self.assertEqual((repo / 'calculator.py').read_text(), fixed)
                    unpack(await session.call_tool('write_file', {'path': 'calculator.py', 'content': buggy}))
                    self.assertTrue(unpack(await session.call_tool('git_apply', {'patch': patch}))['applied'])
                    self.assertTrue(unpack(await session.call_tool('run_tests', {}))['tests_passed'])
                    forbidden = patch.replace('calculator.py', 'test_calculator.py')
                    self.assertTrue((await session.call_tool('git_apply', {'patch': forbidden})).model_dump(by_alias=True)["isError"])
                    self.assertEqual(len((await session.list_tools()).tools), 5)


if __name__ == '__main__':
    unittest.main()
