---
name: pr-description-writer
description: 当任务要求编写 PR 描述、提交说明、变更摘要或总结修复结果时加载；适用于 pull request description、commit summary 和 change summary。
---

# 变更说明

1. 阅读 issue，通过 git_diff 核对真实修改。补丁为空时明确说明，不能声称已经修复。
2. 简述原问题、修改后的行为和修改原因，不添加与当前变更无关的背景。
3. 根据已经获得的 run_tests 结果列出验证情况；没有实际执行时写明未运行。
4. 输出简洁的修改说明、测试结果和尚未解决的问题。只生成文本，不自动提交或发布。
