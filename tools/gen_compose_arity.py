"""Regenerate the fixed-arity ColumnTransformer / make_column_transformer overloads.

Mojo 1.0 cannot store a variadic type pack, so each arity is spelled out once
over the recursive `ColumnCons` list. Run after editing the templates below:
    python tools/gen_compose_arity.py
"""
from pathlib import Path

MAX = 8
SOURCE = Path(__file__).resolve().parents[1] / 'src/moxie/compose/_column_transformer.mojo'
OPTIONS = '''*,
        var remainder: TransformerSpec[RR] = "drop",
        sparse_threshold: Float64 = 0.3,
        n_jobs: Optional[Int] = None,
        var transformer_weights: Optional[Dict[String, Float64]] = None,
        verbose: Bool = False,
        var verbose_feature_names_out: FeatureNamesOutOption = True,'''
MAKE_OPTIONS = '''*,
    var remainder: TransformerSpec[RR] = "drop",
    sparse_threshold: Float64 = 0.3,
    n_jobs: Optional[Int] = None,
    verbose: Bool = False,
    var verbose_feature_names_out: FeatureNamesOutOption = True,'''


def chain(n, args=None):
    out = 'EndColumns[d]'
    for i in range(n, 0, -1):
        out = f'ColumnCons[E{i}, {out}]'
    if args is None:
        return out
    value = 'EndColumns[d]()'
    kinds = ['EndColumns[d]']
    for i in range(n, 0, -1):
        kinds.append(f'ColumnCons[E{i}, {kinds[-1]}]')
        value = f'{kinds[-1]}({args[i - 1]}, {value})'
    return out, value


def constructors():
    blocks = []
    for n in range(MAX + 1):
        params = ''.join(f'E{i}: ColumnComponent, ' for i in range(1, n + 1))
        entries = ''.join(f'var t{i}: ColumnEntry[E{i}, d], ' for i in range(1, n + 1))
        kind, value = chain(n, [f't{i}^' for i in range(1, n + 1)])
        blocks.append(f'''    def __init__[{params}d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
        out self: ColumnTransformer[{kind}, RR],
        {entries}{OPTIONS}
    ):
        """Equivalent of sklearn `ColumnTransformer([...{n} entries...], ...)`."""
        self = ColumnTransformer[{kind}, RR](
            {value}, remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
            transformer_weights=transformer_weights^, verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out^)
''')
    return '\n'.join(blocks)


def makers():
    blocks = []
    for n in range(1, MAX + 1):
        params = ''.join(f'E{i}: ColumnComponent, ' for i in range(1, n + 1))
        entries = ''.join(f'var t{i}: ColumnEntry[E{i}, d], ' for i in range(1, n + 1))
        labels = ', '.join(f'_label(t{i}.transformer)' for i in range(1, n + 1))
        renames = ''.join(f'    t{i}.name = names[{i - 1}]\n' for i in range(n))
        renames = ''.join(f'    t{i}.name = names[{i - 1}]\n' for i in range(1, n + 1))
        kind, value = chain(n, [f't{i}^' for i in range(1, n + 1)])
        blocks.append(f'''def make_column_transformer[{params}d: DType = DType.float64, RR: ColumnComponent = NoTransformer[d]](
    {entries}{MAKE_OPTIONS}
) raises -> ColumnTransformer[{kind}, RR]:
    """Equivalent of sklearn `make_column_transformer` with {n} unnamed `ColumnEntry(transformer, columns)` values."""
    var names = _auto_names([{labels}])
{renames}    return ColumnTransformer[{kind}, RR](
        {value}, remainder=remainder^, sparse_threshold=sparse_threshold, n_jobs=n_jobs, verbose=verbose,
        verbose_feature_names_out=verbose_feature_names_out^)
''')
    return '\n\n'.join(blocks)


def splice(text, tag, body, indent):
    begin = f'{indent}# GENERATED-{tag}-BEGIN (tools/gen_compose_arity.py)\n'
    end = f'{indent}# GENERATED-{tag}-END'
    start = text.index(begin) + len(begin)
    stop = text.index(end)
    return text[:start] + body + '\n' + text[stop:]


text = SOURCE.read_text()
text = splice(text, 'ARITY', constructors(), '    ')
text = splice(text, 'MAKE', makers(), '')
SOURCE.write_text(text)
