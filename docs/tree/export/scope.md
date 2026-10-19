# Export scope

`moxie.tree._export` provides `export_treelite`, a Moxie extension that is not
part of the scikit-learn API. It converts fitted dense decision and extra trees
to Treelite v4 models through the native Balsa codec. The output is
byte-identical to Treelite 4.6.1 `treelite.sklearn.import_model` applied to the
same tree wrapped as a one-tree RandomForest/ExtraTrees ensemble. That reference
always sets `default_left=True`, so NaN inputs follow Treelite's left branch even
where the fitted tree's `missing_go_to_left` sends them right. Predictions for
finite inputs agree with the tree.

`export_text` renders fitted decision and extra trees with numeric rules, class
weights, regression values, custom names and truncation. Categorical rules extend
the reference output with actual category sets and missing/unknown routing.
`export_graphviz` returns native DOT for numeric and categorical trees, including
multiple outputs, weights, names, labels, truncation, color fills, parallel leaves
and HTML labels. `plot_tree` renders standalone SVG with the same labels and colors. See the [family scope](../scope.md) and
[native API](../estimators.md).

`pixi run tree-build` compiles the Moxie-owned tree modules; `pixi run tree-test` and
`pixi run tree-parity` run the integrated family checks, including the Treelite
byte oracle. SCAFFOLD_BASE retains the foundation baseline.
