import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.agent_runtime import model_loop
from src.skill_loader import SkillLoader
from src.subagents import run_subagent


class CodingAgent:
    def __init__(self, *, model=None, base_url=None, max_steps=12, use_skills=True, use_subagents=True):
        self.model = model or os.getenv('CODING_MODEL', 'qwen2.5-coder:7b-instruct')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'http://localhost:11434/v1')
        if urlparse(self.base_url).hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('本任务只允许本机模型服务')
        if type(max_steps) is not int or not 1 <= max_steps <= 30:
            raise ValueError('max_steps 必须在 1–30 之间')
        self.max_steps, self.use_skills, self.use_subagents = max_steps, use_skills, use_subagents

    def run(self, repo_path, issue):
        return asyncio.run(self.run_async(repo_path, issue))

    async def run_async(self, repo_path, issue):
        repo = Path(repo_path).resolve()
        if not repo.is_dir() or not isinstance(issue, str) or not issue.strip():
            raise ValueError('需要有效的仓库目录和非空 issue')
        trace = dict(repo=str(repo), issue=issue, model=self.model, steps=[], subagents=[],
                     skills=[], patch='', tests_passed=False, stop_reason='not_started')
        task = issue + '\n仓库根目录已有文件：' + ', '.join(sorted(p.name for p in repo.iterdir() if p.is_file()))
        params = StdioServerParameters(command=sys.executable,
                 args=['-B', str(ROOT / 'src/mcp_server.py'), '--repo', str(repo)],
                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'))
        async with AsyncOpenAI(base_url=self.base_url, api_key=os.getenv('OPENAI_API_KEY', 'ollama'),
                               timeout=120, max_retries=0) as client:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=90) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    tools = [dict(name=t.name, description=t.description,
                                  input_schema=t.model_dump(by_alias=True)['inputSchema']) for t in listing.tools]

                    async def mcp_call(name, arguments):
                        response = await session.call_tool(name, arguments)
                        data = response.model_dump(by_alias=True)
                        text = '\n'.join(x['text'] for x in data['content'] if x['type'] == 'text')
                        if data.get('isError'):
                            raise RuntimeError(text)
                        return json.loads(text)

                    loader = SkillLoader(ROOT / 'src/skills') if self.use_skills else None
                    instructions = '先读取源码和测试，再修复 issue。修改后必须运行测试。需要帮助可委派子 Agent。'
                    if loader:
                        instructions += '\n可选 Skill 元数据：' + json.dumps(loader.list_skills(), ensure_ascii=False)
                        for skill in loader.match(issue):
                            body = loader.load(skill['name'])
                            trace['skills'].append(dict(name=skill['name'], body=body))
                            instructions += '\n已匹配 Skill ' + skill['name'] + ':\n' + body
                        tools.append(dict(name='load_skill', description='按名称加载 Skill 正文',
                                          input_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}))

                    async def delegate(role, task):
                        if len(trace['subagents']) >= 3:
                            raise ValueError('本轮最多委派 3 次子 Agent')
                        sub = await run_subagent(role, task, client, self.model, tools, mcp_call)
                        trace['subagents'].append(sub)
                        return dict(role=role, summary=sub['summary'] or '子任务未在步数限制内完成，请自行读取和核查。',
                                    stop_reason=sub['stop_reason'])

                    if self.use_subagents:
                        tools.append(dict(name='delegate', description='委派只读审查或测试子 Agent，获得摘要',
                                          input_schema={'type': 'object', 'properties': {'role': {'type': 'string', 'enum': ['reviewer', 'tester']},
                                                         'task': {'type': 'string'}}, 'required': ['role', 'task']}))
                        summary = await delegate('reviewer', task)
                        instructions += '\n只读审查摘要：' + json.dumps(summary, ensure_ascii=False)

                    async def dispatch(name, arguments):
                        if name == 'load_skill' and loader:
                            body = loader.load(arguments['name'])
                            trace['skills'].append(dict(name=arguments['name'], body=body))
                            return {'body': body}
                        if name == 'delegate' and self.use_subagents:
                            return await delegate(**arguments)
                        return await mcp_call(name, arguments)

                    result = await model_loop(client, self.model, task, tools, dispatch,
                                              max_steps=self.max_steps, instructions=instructions)
                    trace.update(result)
                    # 模型的完成声明不能代替真实测试；最后检查当前代码并获取实际补丁。
                    for name in ['run_tests', 'git_diff']:
                        try:
                            observation = await mcp_call(name, {})
                            if name == 'run_tests':
                                trace['tests_passed'] = observation.get('returncode') == 0
                            else:
                                trace['patch'] = observation['patch']
                        except Exception as error:
                            observation = {'error': str(error)}
                        trace['steps'].append(dict(step=len(trace['steps']) + 1, thought='程序最终核验',
                                                   tool_call={'name': name, 'arguments': {}}, observation=observation))
                    trace['total_tokens'] = trace['tokens'] + sum(s['tokens'] for s in trace['subagents'])
        return trace


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='本地 MCP Coding Agent')
    parser.add_argument('--repo', type=Path, default=ROOT / 'data/toy-repo')
    parser.add_argument('--max-steps', type=int, default=12)
    parser.add_argument('--no-skills', action='store_true')
    parser.add_argument('--no-subagents', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'eval/trace.json')
    args = parser.parse_args()
    issue = (args.repo / 'ISSUE.md').read_text(encoding='utf-8')
    result = CodingAgent(max_steps=args.max_steps, use_skills=not args.no_skills,
                         use_subagents=not args.no_subagents).run(str(args.repo), issue)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    args.output.with_suffix('.patch').write_text(result['patch'], encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ['tests_passed', 'stop_reason', 'total_tokens']}, ensure_ascii=False))
    print(f'Trace：{args.output.resolve()}')
    raise SystemExit(0 if result['tests_passed'] else 1)
