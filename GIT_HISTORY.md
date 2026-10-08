# Git 整合记录

2026-10-08 将 `graph_beginner`、`llm-beginner`、`transformers_tasks` 和 `Report` 的当前内容整合到 `Beginner` 根目录的单一 Git 仓库。

整合前的提交：

| 原目录 | 提交 |
| --- | --- |
| `graph_beginner` | `1ea83d6fc138b129521a4fb33b74b9c9ae3f360c` |
| `llm-beginner` | `db0f80bfd7ccdcdf450a8123fe5be9d3ef9879ec` |
| `transformers_tasks` | `c5e0f86b689721931f6e0daecb55692f99f3ec70` |
| `Report` | `8c9e523c1c4609b2fd0e043884f33d95f1658795` |

四个原 `.git` 目录已备份到 `E:\Workspace\Beginner-git-backup-20261008`，其中的 `manifest.json` 记录了整合时排除的文件。Coding Agent 测试数据中的 `toy-repo/.git` 也备份在同一目录下的 `task-6-toy-repo.git`。新仓库跟踪代码、文档及体积较小的实验结果；数据集、模型参数、虚拟环境和缓存留在工作区，不纳入 Git。
