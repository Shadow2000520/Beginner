import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent_runtime import model_loop
from src.subagents import run_subagent
from src.agent import CodingAgent


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.history = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.history.append(copy.deepcopy(kwargs['messages']))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content=response), finish_reason='stop')])


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_recovery_and_completion(self):
        client = FakeClient(['bad', '{"tool":"read_file","arguments":{"path":"missing"}}',
                             '{"done":true,"summary":"无法定位文件"}'])
        async def dispatch(name, arguments):
            raise FileNotFoundError(arguments['path'])
        result = await model_loop(client, 'fake', '任务', [{'name': 'read_file'}], dispatch, max_steps=3)
        self.assertEqual(result['stop_reason'], 'done')
        self.assertIn('error', result['steps'][0]['observation'])
        self.assertIn('FileNotFoundError', client.history[-1][-1]['content'])

    async def test_subagent_tool_isolation(self):
        called = []
        async def dispatch(name, arguments):
            called.append(name)
        client = FakeClient(['{"tool":"write_file","arguments":{}}', '{"done":true,"summary":"只读审查"}'])
        result = await run_subagent('reviewer', '检查代码', client, 'fake',
                                    [{'name': 'read_file'}, {'name': 'write_file'}], dispatch)
        self.assertEqual(called, [])
        self.assertEqual(result['summary'], '只读审查')
        self.assertNotIn('write_file', client.history[0][0]['content'])

    async def test_limits_and_model_errors(self):
        async def dispatch(name, arguments):
            return {}
        result = await model_loop(FakeClient(['bad'] * 2), 'fake', '任务', [], dispatch, max_steps=2)
        self.assertEqual(result['stop_reason'], 'max_steps')
        self.assertEqual(len(result['steps']), 2)
        result = await model_loop(FakeClient([RuntimeError('offline')]), 'fake', '任务', [], dispatch, max_steps=2)
        self.assertEqual(result['stop_reason'], 'model_error')

    def test_local_endpoint(self):
        with self.assertRaises(ValueError):
            CodingAgent(base_url='https://example.com/v1')


if __name__ == '__main__':
    unittest.main()
