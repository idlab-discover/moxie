"""Compare native public signatures with source ASTs from the recorded sklearn revision.

For every name in sklearn.preprocessing.__all__, the first native `__init__` of
the struct (or the first overload of the function) must preserve parameter
names, order, keyword-only boundaries and defaults. Each language-driven
difference must appear in ADAPTATIONS with its documented reason.
"""
import ast
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git', '-C', str(REFERENCE), 'rev-parse', 'HEAD'], text=True).strip() == REVISION

# (public name, parameter) -> reason. Keep in sync with docs/api.md.
ADAPTATIONS = json.loads((ROOT / 'tests/api_adaptations.json').read_text())


def tree(relative):
    source = subprocess.check_output(['git', '-C', str(REFERENCE), 'show', f'{REVISION}:{relative}'], text=True)
    assert (REFERENCE / relative).read_text() == source, f'reference drift: {relative}'
    return ast.parse(source)


def split_top(text):
    parts, depth, quote, current = [], 0, None, ''
    for ch in text:
        if quote:
            current += ch
            if ch == quote:
                quote = None
            continue
        if ch in '"\'':
            quote = ch
        elif ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
            continue
        current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def signature_text(source, start):
    """Text between the outer parentheses of a def beginning at `start`."""
    i = source.index('(', start)
    # Skip a comptime parameter list: def f[...](...)
    bracket = source.find('[', start, i)
    if bracket != -1:
        depth, j = 0, bracket
        while True:
            depth += {'[': 1, ']': -1}.get(source[j], 0)
            if depth == 0:
                break
            j += 1
        i = source.index('(', j)
    depth, j = 0, i
    while True:
        depth += {'(': 1, ')': -1}.get(source[j], 0)
        if depth == 0:
            return source[i + 1:j]
        j += 1


_NAN = re.compile(r'^(Float(32|64)\("nan"\)|nan\[.*\]\(\)|bitcast\[DType\.float64\]\(UInt64\(0x7ff8000000000000\)\))$')


def native_default(text):
    text = text.strip()
    if _NAN.match(text):
        return math.nan
    if re.match(r'^Float(32|64)\("inf"\)$', text):
        return math.inf
    if text.startswith('DType.'):
        return 'np.' + text[len('DType.'):]
    text = re.sub(r'^(Float64|Float32|Int|Scalar\[[^\]]*\])\((.*)\)$', r'\2', text)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ('expression', text)


def native_params(text):
    params, keyword_only = [], False
    for part in split_top(text):
        if part in ('out self', 'self', 'mut self'):
            continue
        if part == '*':
            keyword_only = True
            continue
        name, _, rest = part.partition(':')
        name = name.strip().removeprefix('var ').removeprefix('mut ')
        default = None
        has_default = '=' in rest
        if has_default:
            default = native_default(split_default(rest))
        params.append((name, keyword_only, has_default, default))
    return params


def split_default(rest):
    depth = 0
    for k, ch in enumerate(rest):
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == '=' and depth == 0:
            return rest[k + 1:]
    raise ValueError(rest)


def reference_default(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('int', 'float') and len(node.args) == 1:
        return {'int': int, 'float': float}[node.func.id](ast.literal_eval(node.args[0]))
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'np':
        return math.nan if node.attr == 'nan' else 'np.' + node.attr
    try:
        return ast.literal_eval(node)
    except ValueError:
        return ('expression', ast.unparse(node))


def reference_params(args):
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    params = [(a.arg, False, d is not None, reference_default(d) if d is not None else None)
              for a, d in zip(positional, defaults) if a.arg != 'self']
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        params.append((a.arg, True, d is not None, reference_default(d) if d is not None else None))
    if args.vararg or args.kwarg:
        params.append(('**' + (args.kwarg or args.vararg).arg, True, False, None))
    return params


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and not (a and a[0] == 'expression'):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b) or (isinstance(a, (int, float)) and not isinstance(a, bool)
                                               and isinstance(b, (int, float)) and not isinstance(b, bool) and a == b)


module = tree('sklearn/preprocessing/__init__.py')
public = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == '__all__' for t in n.targets))
exports = json.loads((ROOT / 'EXPORTS.json').read_text())['packages']['moxie.preprocessing']
assert set(public) == {name for name, owner in exports.items() if not owner.endswith('._base')}

errors, used, checked = [], set(), 0
for node in module.body:
    if not isinstance(node, ast.ImportFrom):
        continue
    definitions = {n.name: n for n in tree(node.module.replace('.', '/') + '.py').body
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for imported in node.names:
        name = imported.name
        if name not in public:
            continue
        source = (ROOT.parents[1] / 'src' / (exports[name].replace('.', '/') + '.mojo')).read_text()
        definition = definitions[name]
        if isinstance(definition, ast.ClassDef):
            init = next((n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            reference = reference_params(init.args) if init else []
            struct = re.search(r'^struct ' + name + r'\b', source, re.M)
            end = re.search(r'^(struct |def |@|comptime |trait )', source[struct.end():], re.M)
            block_end = struct.end() + (end.start() if end else len(source))
            first = source.find('    def __init__(out self', struct.end(), block_end)
            assert first != -1, f'{name}: no native constructor'
            native = native_params(signature_text(source, first))
        else:
            reference = reference_params(definition.args)
            first = re.search(r'^def ' + name + r'\b', source, re.M)
            native = native_params(signature_text(source, first.start()))
        native_map = {p[0]: p for p in native}
        reference_names = [p[0] for p in reference]
        for ref in reference:
            key = f'{name}.{ref[0]}'
            got = native_map.get(ref[0])
            if key in ADAPTATIONS:
                used.add(key)
                continue
            if got is None:
                errors.append(f'{key}: missing native parameter')
            elif got[1:3] != ref[1:3] or (ref[2] and not same(got[3], ref[3])):
                errors.append(f'{key}: native {got[1:]} != reference {ref[1:]}')
        order = [p[0] for p in native if p[0] in reference_names]
        if order != [n for n in reference_names if n in native_map]:
            errors.append(f'{name}: parameter order {order} differs from {reference_names}')
        for extra in native:
            key = f'{name}.{extra[0]}'
            if extra[0] not in reference_names:
                if key in ADAPTATIONS:
                    used.add(key)
                else:
                    errors.append(f'{key}: native-only parameter is not a documented adaptation')
        checked += 1
stale = set(ADAPTATIONS) - used
assert not stale, f'unused adaptation entries: {sorted(stale)}'
assert not errors, '\n'.join(errors)
assert checked == len(public) == 29
print(f'PASS 29 public signatures against pinned sklearn ({len(ADAPTATIONS)} documented adaptations)')
