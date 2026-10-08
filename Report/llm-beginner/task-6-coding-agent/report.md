# Task 6：Mini Coding Agent 实验报告

## 目标完成情况

- M1：实现基于 stdio 的 MCP server，提供 read_file、write_file、run_tests、git_diff、git_apply 五个工具，工具枚举自检通过。
- M2：实现 SkillLoader 与 code-review、test-runner、pr-description-writer 三个 Skill，元数据自检通过。加载器先读取 name 和 description，匹配后再加载正文。
- M3：实现主 Agent 循环和独立上下文的 reviewer、tester 子 Agent。本次实际调用 reviewer，完成 toy-repo 的加法修复，三个 pytest 全部通过，测试文件未修改。
- M4：保存主 Agent 与子 Agent 的完整步骤、真实工具返回值、补丁和 tests_passed。主循环本次由步数上限结束，尚未形成正常的主动完成声明。

## 实验设置

使用本地 Ollama 的 qwen2.5-coder:7b-instruct，部署格式为 Q4_K_M。temperature=0，每轮最多输出 1200 个词元；主循环最多 12 轮，子 Agent 最多 4 轮，每次运行最多委派 3 次。仓库工具均经 MCP 客户端调用。

输入为 data/toy-repo/ISSUE.md，要求修复 calculator.add，不允许修改测试文件。评测前从 calculator.py.orig 恢复初始错误代码。Skill 默认启用，按 issue 描述匹配；本次初始加载 test-runner 与 pr-description-writer。主 Agent 只接收 reviewer 的摘要，子 Agent 全文另存于 trace。

## 正式结果

- mcp_server_lists_tools：通过，共 5 个工具。
- skill_loader_metadata：通过，共 3 个 Skill，缺失元数据为 0。
- toy_repo_patch：通过，pytest 输出为 3 passed in 0.01s；trace_has_patch=true，tests_unchanged=true。
- swebench_lite_sample：未下载选做数据，正常跳过。
- 主 Agent：12 轮模型调用，外加 2 次程序最终核验，共 14 条主 trace。
- reviewer：4 轮，读取源码与测试、查看差异后，正确指出加法被写成减法。tester 本次未被调用。
- API 返回的累计词元用量：主 Agent 21,935，reviewer 2,184，合计 24,119。该数值包含各次请求反复发送的上下文，不是生成文本长度。
- stop_reason=max_steps，tests_passed=true，compactions=0。本次未触发历史裁剪。

## 修复过程与实际补丁

主 Agent 第 1 步读取 calculator.py，第 2 步将 add 的返回表达式改为加法，第 3 步运行测试并得到全部通过，第 4 步取得补丁。修改仅涉及一行，factorial 和测试文件保持不变。

```diff
diff --git a/calculator.py b/calculator.py
index 20ddc0a..6bd5739 100644
--- a/calculator.py
+++ b/calculator.py
@@ -1,6 +1,6 @@
 def add(a, b):
     """Return the sum of two numbers."""
-    return a - b
+    return a + b
 
 def factorial(n):
     if n < 0:
```

第 5、7、9、11 步把 pr-description-writer 当成工具调用，被工具校验拒绝；第 6、8、10、12 步反复加载该 Skill，未能纠正概念混淆。主循环达到上限后，程序第 13 步重新测试，第 14 步获取真实补丁，两项均成功。

## 实验观察
本实验完成了 MCP 工具、Skill 加载器和独立上下文子 Agent 的组合，并让本地 Coder 模型修复了 toy-repo。reviewer 读取源码与测试后给出正确诊断，主 Agent 在第 2 步完成修改，第 3 步通过全部测试，说明从问题定位到修改、验证的链路能够工作。但本次也暴露出 Skill 与 Tool 的概念混淆：模型随后尝试直接调用 pr-description-writer，出错后不断重新加载正文，最终由步数上限结束。程序的最终测试和补丁核验保证了结果可信，却没有解决循环中的重复操作。后续应明确 Skill 只是操作指引，并在修复验证完成后引导模型及时输出完成信号。三层能力已接通，但本次仅测试一个简单缺陷，未开展关闭 Skill、关闭 Subagent 或不同量化格式的对照，因此不能据此认定它们提高了成功率或降低了词元消耗。

## 实验产物

- eval/result.json：正式自检结果。
- eval/trace.json：主 Agent、子 Agent、Skill 加载记录与词元统计。
- eval/trace.patch：实际源码修改补丁。
- experiment_trace.md：完整主／子 Agent 调用记录附录。
