import ast
import builtins
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

TOOL_SCHEMA = {'type': 'function', 'function': {
    'name': 'python_sandbox', 'description': '执行可信的短 Python 代码，使用 print 输出结果；允许 math/statistics，限时 3 秒，禁止文件和网络访问接口。',
    'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                   'required': ['code'], 'additionalProperties': False}}}
TIMEOUT = 3
MODULES = {'math', 'statistics'}
BUILTINS = 'print range sum len min max abs round sorted reversed enumerate zip map filter list tuple dict set str int float bool all any isinstance Exception ValueError TypeError ZeroDivisionError'.split()


def validate(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 10000:
        raise ValueError('code 必须是 1–10000 字符的代码')
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.With, ast.AsyncWith, ast.Global, ast.Nonlocal)):
            raise ValueError('不支持类、上下文管理器或全局声明')
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('不允许访问私有名称')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('不允许访问私有属性')
        if isinstance(node, ast.Import) and any(x.name not in MODULES for x in node.names):
            raise ValueError('只能导入 math/statistics')
        if isinstance(node, ast.ImportFrom) and (node.level or node.module not in MODULES or any(x.name.startswith('_') or x.name == '*' for x in node.names)):
            raise ValueError('不允许该导入方式')
    return tree


class LimitedOutput(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > 8000:
            raise ValueError('输出超过 8000 字符')
        return super().write(text)


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in MODULES or level:
        raise ValueError('只能导入 math/statistics')
    return builtins.__import__(name, globals, locals, fromlist, level)


def run(args):
    code = args.get('code')
    validate(code)
    try:
        result = subprocess.run([sys.executable, '-I', '-S', str(Path(__file__).resolve())],
                                input=json.dumps({'code': code}), capture_output=True,
                                text=True, encoding='utf-8', timeout=TIMEOUT)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f'Python 执行超过 {TIMEOUT} 秒') from error
    if result.returncode:
        raise RuntimeError('Python 子进程异常退出：' + result.stderr[:500])
    response = json.loads(result.stdout)
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['output'] or '执行完成，没有 stdout；请用 print 输出需要的结果。'


if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        code = json.load(sys.stdin)['code']
        tree = validate(code)
        namespace = {'__builtins__': {name: getattr(builtins, name) for name in BUILTINS}}
        namespace['__builtins__']['__import__'] = restricted_import
        output = LimitedOutput()
        with contextlib.redirect_stdout(output):
            exec(compile(tree, '<tool>', 'exec'), namespace, namespace)
        response = {'output': output.getvalue()}
    except Exception as error:
        response = {'error': f'{type(error).__name__}: {error}'}
    print(json.dumps(response, ensure_ascii=False))
