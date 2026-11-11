"""Bounded arithmetic parser; never executes Python or consults answer labels."""
from __future__ import annotations

import json
import math
import re

EXECUTOR_VERSION = '2-bounded-expressions'
_TOKEN = re.compile(r'''\s*(?:(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?)|(?P<ref>\#\d+)|(?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<symbol>[()+*/,\-])|(?P<raw>[^\s(),+*/\-]+))''')


def execute_expression(program, table, number):
    text = str(program).strip()
    # Only remove complete transport wrappers, never arbitrary prose or suffixes.
    text = re.sub(r'\s*\[\[\s*##\s*completed\s*##\s*\]\]\s*$', '', text).strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```[^\n]*\n', '', text)[:-3].strip()
    if not text or len(text) > 8192:
        raise ValueError('format: empty program or program exceeds 8192 characters')
    tokens, cursor = [], 0
    while cursor < len(text):
        if not text[cursor:].strip():
            break
        match = _TOKEN.match(text, cursor)
        if not match:
            raise ValueError(f'format: unsupported character at position {cursor}')
        tokens.append((match.lastgroup, match.group(match.lastgroup), match.start(match.lastgroup), match.end()))
        cursor = match.end()
        if len(tokens) > 512:
            raise ValueError('format: program exceeds 512 tokens')
    pos, operations, values = 0, 0, []

    def peek():
        return tokens[pos][1] if pos < len(tokens) else None

    def take(expected=None):
        nonlocal pos
        if pos >= len(tokens) or (expected is not None and peek() != expected):
            raise ValueError(f'format: expected {expected or "an expression"}')
        token = tokens[pos]
        pos += 1
        return token

    def finite(value):
        if isinstance(value, bool) or isinstance(value, complex) or not math.isfinite(value):
            raise ValueError('non-finite or non-numeric result')
        return value

    def apply(op, a, b):
        nonlocal operations
        operations += 1
        if operations > 32:
            raise ValueError('format: at most 32 arithmetic operations')
        if isinstance(a, str) or isinstance(b, str):
            raise ValueError('cannot use a comparison as a numeric argument')
        if op == 'exp' and (abs(b) > 16 or abs(a) > 1e16):
            raise ValueError('exponent outside demonstration limits')
        if op == 'greater':
            return 'yes' if a > b else 'no'
        actions = {'add': lambda: a+b, 'subtract': lambda: a-b,
                   'multiply': lambda: a*b, 'divide': lambda: a/b,
                   'exp': lambda: a**b}
        if op not in actions:
            raise ValueError(f'unsupported operation: {op}')
        return finite(actions[op]())

    def table_call(op):
        nonlocal pos, operations
        # The first table argument is a literal row name, not a computed value.
        start = pos
        while peek() not in (None, ',', '(', ')'):
            pos += 1
        if pos == start or peek() != ',':
            raise ValueError('format: table operation requires a literal row name and none')
        row_name = text[tokens[start][2]:tokens[pos-1][3]].strip()
        if tokens[start][0] == 'string' and pos == start+1:
            row_name = string_value(row_name)
        elif row_name.startswith('#'):
            raise ValueError('table row must be a literal name, not a step reference')
        take(',')
        if take()[1].lower() != 'none':
            raise ValueError('table operations aggregate a row; second argument must be none, not a column/year')
        take(')')
        rows = [r[1:] for r in table if str(r[0]).strip() == row_name]
        if len(rows) != 1:
            raise ValueError(f'table row not found or ambiguous: {row_name}')
        nums = [number(x) for x in rows[0]]
        if not nums:
            raise ValueError('empty table row')
        operations += 1
        if operations > 32:
            raise ValueError('format: at most 32 arithmetic operations')
        actions = {'table_sum': lambda: sum(nums), 'table_average': lambda: sum(nums)/len(nums),
                   'table_min': lambda: min(nums), 'table_max': lambda: max(nums)}
        if op not in actions:
            raise ValueError(f'unsupported operation: {op}')
        return finite(actions[op]())

    def string_value(value):
        if value.startswith('"'):
            return json.loads(value)
        return value[1:-1].replace("\\'", "'").replace('\\\\', '\\')

    def expression(min_precedence=0, depth=0):
        if depth > 32:
            raise ValueError('format: expression nesting exceeds 32 levels')
        kind, token, _, _ = take()
        if token in ('+', '-'):
            result = expression(30, depth+1)
            if isinstance(result, str):
                raise ValueError('unary sign requires a number')
            value = finite(result if token == '+' else -result)
        elif token == '(':
            value = expression(0, depth+1)
            take(')')
        elif kind == 'number':
            value = number(token)
        elif kind == 'string':
            value = number(string_value(token))
        elif kind == 'ref':
            index = int(token[1:])
            if not 0 <= index < len(values):
                raise ValueError('reference must point to an earlier top-level step')
            value = values[index]
        elif kind == 'name' and token.startswith('const_'):
            value = number(token)
        elif kind == 'name':
            take('(')
            if token.startswith('table_'):
                value = table_call(token)
            else:
                a = expression(0, depth+1)
                take(',')
                b = expression(0, depth+1)
                take(')')
                value = apply(token, a, b)
        else:
            raise ValueError('format: expected a number, operation or earlier step reference')
        while peek() in ('+', '-', '*', '/'):
            op = peek()
            precedence = 10 if op in ('+', '-') else 20
            if precedence < min_precedence:
                break
            take()
            right = expression(precedence+1, depth+1)
            value = apply({'+':'add','-':'subtract','*':'multiply','/':'divide'}[op], value, right)
        return value

    while pos < len(tokens):
        if peek() == 'EOF' and values:
            take('EOF')
            if pos != len(tokens):
                raise ValueError('format: unexpected text after EOF')
            break
        values.append(expression())
        if peek() == ',':
            take(',')
            if pos == len(tokens):
                raise ValueError('format: trailing comma without a step')
        elif peek() == 'EOF':
            take('EOF')
            if pos != len(tokens):
                raise ValueError('format: unexpected text after EOF')
        elif pos != len(tokens):
            raise ValueError('format: expected comma between calculation steps')
    if not values:
        raise ValueError('format: empty program')
    return values[-1] if isinstance(values[-1], str) else round(values[-1], 5)
