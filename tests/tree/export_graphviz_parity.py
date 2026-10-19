"""Exact DOT comparisons against the source-built pinned sklearn oracle."""
from pathlib import Path
import json
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "builds/tree/matplotlib"))
import xml.etree.ElementTree as ET
from sklearn.tree._export import _MPLTreeExporter
import subprocess
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor, export_graphviz

ROOT = Path(__file__).resolve().parents[2]

def check_svg(svg, model, options, fontsize):
    ns = {'s': 'http://www.w3.org/2000/svg'}
    root = ET.fromstring(svg)
    _, _, width, height = map(float, root.attrib['viewBox'].split())
    assert np.isfinite([width, height]).all() and min(width, height) > 0
    exporter = _MPLTreeExporter(**{k: v for k, v in options.items() if k not in ('rotate', 'leaves_parallel', 'special_characters')})
    expected = {}
    stack = [(0, 0, None)]
    edges = set()
    while stack:
        node, depth, parent = stack.pop()
        truncated = options['max_depth'] is not None and depth > options['max_depth']
        text = '(...)' if truncated else exporter.node_to_str(model.tree_, node, model.criterion)
        color = '#c0c0c0' if truncated else exporter.get_fill_color(model.tree_, node) if options['filled'] else '#ffffff'
        expected[node] = (text, color)
        if parent is not None:
            edges.add((parent, node))
        if not truncated and model.tree_.children_left[node] != -1:
            stack.extend([(model.tree_.children_left[node], depth + 1, node), (model.tree_.children_right[node], depth + 1, node)])
    found = {}
    boxes = []
    for group in root.findall('s:g', ns):
        node = int(group.attrib['data-node-id'])
        text = '\n'.join(t.text or '' for t in group.findall('s:text', ns))
        rect = group.find('s:rect', ns)
        assert node not in found
        found[node] = (text, rect.attrib['fill'])
        assert all(float(t.attrib['font-size']) == fontsize for t in group.findall('s:text', ns))
        x, y, w, h = [float(rect.attrib[k]) for k in ('x', 'y', 'width', 'height')]
        assert 0 <= x < x + w <= width and 0 <= y < y + h <= height
        for bx, by, bw, bh in boxes:
            assert x + w <= bx or bx + bw <= x or y + h <= by or by + bh <= y
        boxes.append((x, y, w, h))
    assert found == expected, (found, expected)
    actual_edges = {(int(p.attrib['data-parent']), int(p.attrib['data-child'])) for p in root.findall('s:path', ns)}
    assert actual_edges == edges


def main():
    assert 'builds/tree/oracle-env' in sklearn.__file__
    revision = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
    expected = subprocess.check_output(['git', '-C', str(ROOT.parent / 'scikit-learn'), 'show', f'{revision}:sklearn/tree/_export.py'])
    assert (ROOT / 'builds/tree/reference/sklearn/tree/_export.py').read_bytes() == expected
    output = subprocess.check_output([ROOT / 'builds/tree/export_graphviz_driver'], text=True)
    actual = {}
    svgs = {}
    for block in output.split('REPORT ')[1:]:
        key, report = block.split('\n', 1)
        dot, svg = report.split("\nSVG\n", 1)
        actual[tuple(map(int, key.split()))] = dot
        svgs[tuple(map(int, key.split()))] = svg
    failures = []
    checks = 0
    for tag, cls in enumerate((DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier)):
        classifier = tag % 2 == 1
        for case in range(20):
            n = 16 + case
            outputs = 2 if case % 3 == 0 else 1
            X = np.array([[((i * 3 + case) % 13) / 4, (i % 7) / 4] for i in range(n)], dtype=np.float32)
            y = np.array([[(i + i // 3 + k) % (1 if case == 1 else 7 if case == 19 else 3) if classifier else ((i * 7 + case + k) % 13) / 4 for k in range(outputs)] for i in range(n)], dtype=float)
            w = np.array([(1 + i % 3) / 4 for i in range(n)])
            criterion = ('gini' if classifier else 'squared_error')
            if case % 4 == 2:
                criterion = 'entropy' if classifier else 'absolute_error'
            elif case % 4 == 3:
                criterion = 'log_loss' if classifier else 'poisson'
            model = cls(criterion=criterion, random_state=case, max_depth=14 if case == 19 else 4,
                        max_leaf_nodes=5 if case % 4 == 2 else None).fit(X, y, sample_weight=w)
            for mode in range(16):
                kw = dict(feature_names=['height', 'colour intensity'] if mode == 1 else None, class_names=['class ' + str(c) for c in range(int(model.tree_.n_classes[0]))] if classifier and outputs == 1 and case != 1 and mode == 1 else None, max_depth=0 if mode == 2 or mode == 9 or mode == 13 else None, precision=3, fill_colors=['#ff0000' if c == 0 else '#00ff00' if c == 1 else '#0000ff' for c in range(int(model.tree_.n_classes[0]))] if mode == 14 else None, label='root' if mode == 3 else 'none' if mode == 4 else 'all', impurity=mode != 5, node_ids=mode == 6, proportion=mode == 7, rounded=mode == 6 or mode == 12, rotate=mode == 7 or mode == 12, filled=mode >= 8, leaves_parallel=mode == 10 or mode == 12, special_characters=mode == 11 or mode == 12 or mode == 13)
                want = export_graphviz(model, **kw)
                got = actual[tag, case, mode].rstrip("\n")
                check_svg(svgs[tag, case, mode], model, kw, 10 if mode == 15 else 14)
                checks += 1
                if got != want:
                    failures.append(dict(key=[tag, case, mode], got=got, want=want))
    dest = ROOT / 'docs/private/tree/export_graphviz'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'native.txt').write_text(output)
    (dest / 'results.json').write_text(json.dumps(dict(checks=checks, svg_checks=checks, failures=failures), indent=2))
    print(f'{checks} exact DOT comparisons and {checks} SVG structure/label/color/layout checks; {len(failures)} mismatches')
    assert not failures, failures[:2]

if __name__ == '__main__':
    main()
