import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import ReActAgent, parse_response


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.history = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.history.append(copy.deepcopy(kwargs['messages']))
        text = next(self.responses)
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')])


class AgentTests(unittest.TestCase):
    def test_json_fences(self):
        parsed = parse_response('Thought: 计算\nAction: calculator\nAction Input:\n```json\n{"expression":"2+3"}\n```')
        self.assertEqual(parsed['action_input'], {'expression': '2+3'})
        for text in ['Action: calculator\nAction Input: []', 'Action: calculator\nFinal Answer: 5',
                     'Action: calculator\nAction Input: {}\nObservation: 5']:
            with self.assertRaises(ValueError):
                parse_response(text)

    def test_recovery_and_history(self):
        client = FakeClient(['格式错误', 'Action: calculator\nAction Input: {"expression":"1/0"}',
                             'Action: calculator\nAction Input: {"expression":"2+3"}', 'Final Answer: 5'])
        trace = ReActAgent(client=client).run('计算')
        self.assertTrue(trace['success'])
        self.assertEqual(trace['final_answer'], '5')
        self.assertIn('解析失败', trace['steps'][0]['error'])
        self.assertIn('ZeroDivisionError', trace['steps'][1]['error'])
        self.assertEqual(trace['steps'][2]['observation'], '5')
        self.assertTrue(any('ZeroDivisionError' in x['content'] for x in client.history[-1]))

    def test_limits_and_service_failure(self):
        trace = ReActAgent(client=FakeClient(['bad'] * 2), max_steps=2).run('任务')
        self.assertFalse(trace['success'])
        self.assertEqual(trace['stop_reason'], 'max_steps')
        self.assertEqual(len(trace['steps']), 2)
        trace = ReActAgent(client=FakeClient([RuntimeError('offline')])).run('任务')
        self.assertEqual(trace['stop_reason'], 'model_error')
        self.assertFalse(trace['success'])

    def test_unknown_tool_and_bad_arguments(self):
        client = FakeClient(['Action: unknown\nAction Input: {}',
                             'Action: calculator\nAction Input: {"code":"2"}', 'Final Answer: 无法完成'])
        trace = ReActAgent(client=client).run('任务')
        self.assertIn('未知工具', trace['steps'][0]['observation'])
        self.assertIn('参数不匹配', trace['steps'][1]['observation'])

    def test_inject_once_per_run(self):
        action = 'Action: calculator\nAction Input: {"expression":"2+3"}'
        agent = ReActAgent(client=FakeClient([action, action, 'Final Answer: 5'] * 2), inject_error=True)
        for _ in range(2):
            trace = agent.run('计算')
            self.assertIn('模拟', trace['steps'][0]['observation'])
            self.assertEqual(trace['steps'][1]['observation'], '5')


if __name__ == '__main__':
    unittest.main()
