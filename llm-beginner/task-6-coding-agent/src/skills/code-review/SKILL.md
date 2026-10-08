---
name: code-review
description: 当任务要求代码审查、检查补丁、定位缺陷或分析 bug 时加载；适用于 code review、patch review 和 bug diagnosis。
---

# 代码审查

1. 通过 read_file 阅读 issue、相关源码和测试，区分预期行为与实际行为。
2. 沿输入、边界条件和返回值定位缺陷。必要时调用 run_tests，依据实际失败信息判断。
3. 只建议与 issue 相关的最小修改，不修改测试、原始快照和测试配置。
4. 修改后用 git_diff 检查补丁范围，确认没有无关变更。
5. 输出具体文件、问题原因和建议；没有证据时说明不确定，不编造测试结果。
