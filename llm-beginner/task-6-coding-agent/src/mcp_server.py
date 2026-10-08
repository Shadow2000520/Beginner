import argparse
import asyncio
from functools import wraps
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp.exceptions import ToolError

ROOT = Path(__file__).resolve().parents[1]


class RepoTools:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        if not self.repo.is_dir():
            raise ValueError('目标仓库目录不存在')

    def path(self, name, writable=False):
        if not name or '\x00' in name or ':' in name or Path(name).is_absolute() or PureWindowsPath(name).drive:
            raise ValueError('只接受仓库内的相对路径')
        parts = name.replace('\\', '/').split('/')
        if '..' in parts or any(p.lower() == '.git' for p in parts):
            raise ValueError('拒绝路径穿越和 .git 访问')
        target = (self.repo / name).resolve()
        if not target.is_relative_to(self.repo):
            raise ValueError('路径超出目标仓库')
        if any(p.lower() == '.git' for p in target.relative_to(self.repo).parts):
            raise ValueError('禁止访问 Git 元数据')
        if writable:
            lowered = [p.lower() for p in target.relative_to(self.repo).parts]
            filename = target.name.lower()
            if (any(p in ('tests', 'test', '__pycache__', '.pytest_cache') for p in lowered)
                    or filename.startswith('test_') or filename.endswith(('_test.py', '.orig'))
                    or filename in ('conftest.py', 'pytest.ini', 'pyproject.toml', 'setup.cfg', 'setup.py', 'issue.md')):
                raise ValueError('禁止修改测试、基准快照、issue 或测试配置')
        return target

    def process(self, command, timeout=60):
        env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
                   PYTEST_ADDOPTS='', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        result = subprocess.run(command, cwd=self.repo, env=env, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout)
        return result

    def git_command(self, *args):
        if not (self.repo / '.git').exists():
            raise ValueError('目标目录必须有独立的 .git，避免操作外层仓库')
        # 允许当前明确指定的工作仓库由另一个本机账户创建，不改全局配置。
        return ['git', '-c', f'safe.directory={self.repo.as_posix()}',
                '-c', 'core.quotepath=false', *args]

    def git(self, *args):
        return self.process(self.git_command(*args))

    def read_file(self, path):
        target = self.path(path)
        if target.stat().st_size > 1024 * 1024:
            raise ValueError('文件超过 1 MiB，请缩小读取范围')
        return {'path': path, 'content': target.read_text(encoding='utf-8')}

    def write_file(self, path, content):
        target = self.path(path, writable=True)
        if len(content.encode('utf-8')) > 1024 * 1024:
            raise ValueError('写入内容超过 1 MiB')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8', newline='\n')
        return {'path': path, 'written': True}

    def run_tests(self):
        result = self.process([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider'])
        return {'tests_passed': result.returncode == 0, 'returncode': result.returncode,
                'stdout': result.stdout[-16000:], 'stderr': result.stderr[-4000:]}

    def git_diff(self):
        result = self.git('diff', '--no-ext-diff', '--no-textconv', '--', '.')
        if result.returncode:
            raise RuntimeError(result.stderr)
        return {'patch': result.stdout}

    def git_apply(self, patch):
        if not patch.strip() or len(patch.encode('utf-8')) > 1024 * 1024:
            raise ValueError('patch 必须为非空且不超过 1 MiB 的文本补丁')
        if re.search(r'^(?:new file mode|deleted file mode|old mode|new mode|rename |copy |GIT binary patch)', patch, re.M):
            raise ValueError('只支持已有文本文件的内容修改')
        old = re.findall(r'^--- a/(.+)$', patch, re.M)
        new = re.findall(r'^\+\+\+ b/(.+)$', patch, re.M)
        if not old or old != new:
            raise ValueError('需要标准 a/、b/ 路径且不能改名或删除文件')
        for name in old:
            target = self.path(name.rstrip('\r'), writable=True)
            if not target.is_file():
                raise ValueError('补丁只能修改已有文件')
        # Git 默认整份补丁成功才应用；先检查，失败时不写入任何文件。
        for check in [True, False]:
            args = ['apply', '--whitespace=nowarn'] + (['--check'] if check else []) + ['-']
            if not (self.repo / '.git').exists():
                raise ValueError('目标仓库缺少 .git')
            env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
            result = subprocess.run(self.git_command(*args), input=patch.encode('utf-8'), cwd=self.repo, env=env,
                                    capture_output=True, timeout=30)
            if result.returncode:
                raise ValueError(result.stderr.decode('utf-8', errors='replace').strip())
        return {'applied': True, 'files': old}


def create_server(repo):
    backend = RepoTools(repo)
    server = MCPServer('mini-coding-tools', log_level='WARNING')

    def tool(function):
        @wraps(function)
        def guarded(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as error:
                raise ToolError(f'{type(error).__name__}: {error}') from error
        return server.tool()(guarded)

    @tool
    def read_file(path: str) -> dict:
        """读取目标仓库内 UTF-8 文件；path 必须为相对路径。"""
        return backend.read_file(path)

    @tool
    def write_file(path: str, content: str) -> dict:
        """写入仓库内源码，禁止修改测试、配置与原始快照。"""
        return backend.write_file(path, content)

    @tool
    def run_tests() -> dict:
        """在目标仓库执行 python -m pytest，最多 60 秒，返回真实退出码和输出。"""
        return backend.run_tests()

    @tool
    def git_diff() -> dict:
        """返回目标仓库中已跟踪文件的未暂存修改补丁。"""
        return backend.git_diff()

    @tool
    def git_apply(patch: str) -> dict:
        """检查并应用已有源码文件的 unified diff；禁止改测试和仓库外文件。"""
        return backend.git_apply(patch)

    return server


def list_tools():
    server = create_server(ROOT)
    tools = asyncio.run(server.list_tools())
    return [dict(name=t.name, description=t.description, input_schema=t.model_dump(by_alias=True)["inputSchema"]) for t in tools]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='限定单个仓库的 MCP stdio server')
    parser.add_argument('--repo', type=Path, default=ROOT / 'data/toy-repo')
    args = parser.parse_args()
    create_server(args.repo).run(transport='stdio')
