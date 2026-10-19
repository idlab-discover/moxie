# Public API status

The assembled `moxie.tree` namespace exports implemented dense decision-tree and
extra-tree estimators from Moxie. See
[estimators.md](estimators.md) for native types and adaptations, and
[scope.md](scope.md) for supported modes and remaining work.

`plot_tree`, `export_text`, `export_graphviz`, and `export_treelite` come from
`moxie.tree._export`. Historical placeholder declarations are removed.
DOT rendering supports the documented subset of Graphviz options.

Compile with `pixi run build`. Dependency assembly generates a single namespace
under `builds/deps` from the current local sources, verifying explicit module
and export ownership. Do not overlay local sources or add competing package roots
to bypass the manifest. When moving modules or exports, update Moxie's root `DEPENDENCIES.json`.
