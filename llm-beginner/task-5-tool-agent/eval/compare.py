import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time

ROOT = Path(__file__).resolve().parents[1]
CHECKS = runpy.run_path(str(ROOT / 'eval/run.py'))
from src.agent import ReActAgent, make_prompt, TOOLS
from _eval_harness import setup_console


def summarize(result):
    details = result['details']
    traces = [item.get('trace', {}) for item in details]
    injected = [item for item in details if any(step.get('injected_error')
                for step in item.get('trace', {}).get('steps', []))]
    recovered = sum(item['success'] and item['trace'].get('stop_reason') == 'final_answer'
                    for item in injected)
    return dict(rate=result['rate'], passed=result['pass'], n=result['n'],
                mean_steps=round(sum(len(t.get('steps', [])) for t in traces) / len(details), 2),
                tool_calls=sum(bool(s.get('action')) for t in traces for s in t.get('steps', [])),
                error_steps=sum(bool(s.get('error')) for t in traces for s in t.get('steps', [])),
                expected_tools_rate=round(sum(d.get('used_expected_tools') is True for d in details) / len(details), 3),
                injected_tasks=len(injected), recovered_tasks=recovered,
                recovery_rate=round(recovered / len(injected), 3) if injected else None)


if __name__ == '__main__':
    setup_console()
    parser = argparse.ArgumentParser(description='10 题正式评测、提示词对照和一次故障注入')
    parser.add_argument('--max-steps', type=int, default=8)
    parser.add_argument('--model')
    args = parser.parse_args()
    tasks = json.loads((ROOT / 'data/tasks.json').read_text(encoding='utf-8'))
    if len(tasks) != 10 or len({t['id'] for t in tasks}) != 10:
        parser.error('需要 10 道 id 唯一的评测题，请先运行 data/download.py')
    output = dict(status='running', tasks_sha256=hashlib.sha256((ROOT / 'data/tasks.json').read_bytes()).hexdigest(),
                  max_steps=args.max_steps, temperature=0, runs=[])
    path = ROOT / 'eval/comparison.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    tools_result = CHECKS['test_tools_individual']()
    print(json.dumps(tools_result, ensure_ascii=False), flush=True)
    for name, few_shot, inject_error in [('baseline', True, False), ('no_example', False, False),
                                         ('injected_error', True, True)]:
        agent = ReActAgent(model=args.model, max_steps=args.max_steps,
                           few_shot=few_shot, inject_error=inject_error)
        print(f'开始 {name}：10 道题……', flush=True)
        started = time.perf_counter()
        result = CHECKS['test_multi_tool_success_rate'](agent)
        if 'details' not in result:
            raise RuntimeError(str(result))
        run = dict(name=name, model=agent.model, base_url=str(agent.client.base_url),
                   prompt=make_prompt(TOOLS, few_shot), inject_error=inject_error,
                   elapsed_seconds=round(time.perf_counter() - started, 2),
                   summary=summarize(result), result=result)
        output['runs'].append(run)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(name=name, **run['summary']), ensure_ascii=False), flush=True)
    output['status'] = 'complete'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    recovery = output['runs'][2]['summary']
    recovery_result = dict(test='error_recovery',
                           **{'pass': recovery['injected_tasks'] == 10 and recovery['recovery_rate'] > .4},
                           injected_tasks=recovery['injected_tasks'], recovered_tasks=recovery['recovered_tasks'],
                           rate=recovery['recovery_rate'], threshold=.4)
    formal = [tools_result, output['runs'][0]['result'], recovery_result]
    (ROOT / 'eval/result.json').write_text(json.dumps(formal, ensure_ascii=False, indent=2), encoding='utf-8')
    print('结果已覆盖保存：eval/comparison.json、eval/result.json；包含三组全部 trace。')
    raise SystemExit(0 if tools_result['pass'] and formal[1]['pass'] else 1)
