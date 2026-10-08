from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.compare import CHECKS, summarize
from src.agent import TOOLS, make_prompt


class EvaluationTests(unittest.TestCase):
    def test_prompt_control(self):
        baseline = make_prompt(TOOLS)
        without = make_prompt(TOOLS, False)
        self.assertEqual(without, baseline.split('\n示例：', 1)[0])
        self.assertIn('工具定义', without)

    def test_self_report_not_used(self):
        class FakeAgent:
            def run(self, task):
                return dict(final_answer='', success=True, steps=[])
        result = CHECKS['test_multi_tool_success_rate'](FakeAgent())
        self.assertEqual(result['rate'], 0)
        self.assertFalse(result['pass'])
        self.assertIn('trace', result['details'][0])

    def test_injection_denominator(self):
        rows = [dict(success=True, trace=dict(stop_reason='final_answer', steps=[dict(injected_error=True, error='fault')])),
                dict(success=False, trace=dict(stop_reason='max_steps', steps=[dict(injected_error=True)])),
                dict(success=True, trace=dict(stop_reason='final_answer', steps=[]))]
        result = summarize(dict(details=rows, rate=.667, n=3, **{'pass': True}))
        self.assertEqual(result['injected_tasks'], 2)
        self.assertEqual(result['recovered_tasks'], 1)
        self.assertEqual(result['recovery_rate'], .5)


if __name__ == '__main__':
    unittest.main()
