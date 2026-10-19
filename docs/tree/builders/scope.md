# Dense scope

DepthFirstTreeBuilder implements the family TreeBuilder trait and selects depth-first or best-first growth from BuilderConfig.max_leaf_nodes. The same owner implements weakest-link pruning and dense PruningPath results.

See the [family scope](../scope.md) for supported modes, explicit dense
gaps and sparse deferrals, and [native API](../estimators.md) for types.
Sparse inputs and CSR decision paths are excluded; they are not densified.
Categorical training and absolute-error regression remain unfinished.

`pixi run tree-build` compiles the Moxie-owned tree modules. `pixi run tree-test` and
`pixi run tree-parity` compile them and invoke the integrated tree contract and
sklearn structural tests. Root `pixi run tree-prepare-oracle` prepares the exact
sklearn source oracle. All generated outputs remain under the family.

SCAFFOLD_BASE retains the foundation baseline; family records are adopted
explicitly through DEPENDENCIES.json. No baseline tag is moved.
