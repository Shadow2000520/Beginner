import ast
import math
import operator

TOOL_SCHEMA = {'type': 'function', 'function': {
    'name': 'calculator', 'description': '计算算术表达式，支持 + - * / // % **、sqrt/log/sin/cos 等数学函数及 pi/e。',
    'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}},
                   'required': ['expression'], 'additionalProperties': False}}}

FUNCTIONS = {name: getattr(math, name) for name in
             ['sqrt', 'log', 'log10', 'exp', 'sin', 'cos', 'tan', 'ceil', 'floor']}
FUNCTIONS.update(abs=abs, round=round)
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
             ast.Mod: operator.mod, ast.Pow: operator.pow}


def calculate(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.Name) and node.id in ('pi', 'e'):
        value = getattr(math, node.id)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = calculate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = calculate(node.left), calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError('指数绝对值不能超过 1000')
        value = OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        value = FUNCTIONS[node.func.id](*(calculate(arg) for arg in node.args))
    else:
        raise ValueError('只允许算术表达式和白名单数学函数')
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError('计算结果超出允许范围')
    return value


def run(args):
    expression = args.get('expression')
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 1000:
        raise ValueError('expression 必须是 1–1000 字符的表达式')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 200:
        raise ValueError('表达式过于复杂')
    return str(calculate(tree.body))
