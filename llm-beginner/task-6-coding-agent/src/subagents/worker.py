from src.agent_runtime import model_loop

ROLES = {
    'reviewer': ('阅读相关源码和测试，诊断 issue，给出修改建议；不修改文件。', {'read_file', 'git_diff'}),
    'tester': ('运行测试并总结真实失败原因或通过情况；不修改文件。', {'read_file', 'run_tests', 'git_diff'}),
}


async def run_subagent(role, task, client, model, tools, dispatch, max_steps=4):
    if role not in ROLES:
        raise ValueError('子 Agent 只能是 reviewer 或 tester')
    instructions, allowed = ROLES[role]
    subset = [tool for tool in tools if tool['name'] in allowed]

    async def call(name, arguments):
        if name not in allowed:
            raise ValueError('子 Agent 无权调用该工具')
        return await dispatch(name, arguments)

    trace = await model_loop(client, model, task, subset, call,
                             max_steps=max_steps, instructions=instructions, final_summary=True)
    trace['role'] = role
    return trace
