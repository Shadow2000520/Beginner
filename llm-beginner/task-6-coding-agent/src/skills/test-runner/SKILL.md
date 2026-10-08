---
name: test-runner
description: 当任务要求运行测试、修复测试失败、修复缺陷或验证修改时加载；适用于 pytest、test failure、debug 和 regression test。
---

# 测试运行与失败诊断

1. 阅读 issue 和测试，明确断言要求；只能修复源码，不得改测试来使结果通过。
2. 调用 run_tests，记录真实退出码、失败断言与报错位置。工具异常不等于测试失败。
3. 阅读相关实现，说明失败原因，再通过 write_file 或 git_apply 做最小修改。
4. 再次调用 run_tests。失败则根据新的 Observation 继续诊断；不重复无效操作。
5. 用 git_diff 获取实际补丁。只有最新源码的测试确实通过，才能汇报 tests_passed；超时或步数耗尽要说明未完成。
