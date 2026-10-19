# Decision trees and extra trees

Moxie owns all four estimators in `src/moxie/tree`: `DecisionTreeClassifier`,
`DecisionTreeRegressor`, `ExtraTreeClassifier`, and `ExtraTreeRegressor`.
Import them and `export_text`, `export_graphviz`, `plot_tree`, or `export_treelite`
from `moxie.tree`. See [the estimator guide](estimators.md), [scope](scope.md),
and [engine contracts](plumbing.md). Run `examples/tree.mojo` for a complete fit.

The implementations moved from moxie-tree without numerical changes. Storage,
criteria, splitting, builders, pruning and exports share one set of modules.
`TreeParams`, `TreeInputEncoder`, and `PreparedTreeInput` are public records;
forests continue to own bootstrap sampling, aggregation and OOB calculations.

Run from the Moxie root:

```sh
pixi run tree-build
pixi run tree-test
pixi run tree-prepare-oracle
pixi run tree-parity
```

The root `build`, `test`, and `parity` commands also include these suites.
The oracle compiles scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`, with source provenance checks.
Treelite 4.6.1 and Matplotlib 3.11.2 are validation-only dependencies. Runtime export uses the external
Balsa native codec at `../../../nvidiads/balsa`, with its license retained under
`LICENSES/tree-Balsa.txt`. No tree build depends on the retired sibling.

Probabilities and learned class domains remain per-output lists, even for one
output. Int64 class labels retain exact identity above 2**53. Regression uses
Float64. Failed fitting preserves the previously fitted tree. Absolute error
rejects missing or categorical features and undefined nonpositive-weight medians.
Sparse input and CSR `decision_path`, arbitrary Python object labels, Python
reflection/dataframes and Matplotlib objects remain unsupported. `plot_tree`
returns native SVG. No new persistence format or forest algorithm is included.
