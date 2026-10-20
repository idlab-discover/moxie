# Verification notebooks

Each notebook replays an official scikit-learn gallery example next to the same
scenario in Moxie, so results can be compared side by side:

1. Python cells run the example with the pinned scikit-learn reference
   (`4358542a1bd39af2e7e1cbace8b9fed067473c5b`).
2. `%%mojo <name>` cells hold complete Mojo programs that run the same scenario
   through Moxie's public API. Learning and prediction happen in Mojo.
3. Comparison cells show both results and record parity checks with justified
   tolerances, plus explicit `GAP` notes for unsupported steps.

Notebooks live under `notebooks/<family>/` and are committed with their
outputs. Start new ones from [`_support/TEMPLATE.ipynb`](_support/TEMPLATE.ipynb).

## Running

```sh
pixi run -e notebooks notebooks                    # execute every notebook in place
pixi run -e notebooks notebooks notebooks/tree     # one family or file
pixi run -e notebooks notebooks-lab                # browse interactively
```

The `notebooks` environment adds Jupyter tooling to the default solve group,
so Mojo, MAX, NumPy and scikit-learn versions match the default environment.

## How `%%mojo` works

`_support/moxie_nb.py` is an IPython extension. A `%%mojo <name>` cell:

1. refreshes the Moxie assembly in `builds/deps` (same tool as `pixi run build`);
2. compiles the cell with `mojo build -O3 --fp-mode=contract=off`, the
   repository's compiler limits, and `-I builds/deps -I notebooks/_support`
   (rebuilds only when the source or the assembly changed);
3. runs the binary and prints its output;
4. loads every array it saved into `nb.results[<name>]`.

Data crosses the boundary as `.npy` files under `builds/notebooks/<key>/`:
`nb.share(X=X)` in Python, `load("X")` / `save("pred", array)` from
`_support/moxie_nb_io.mojo` in Mojo (also `load_ints`, `load_strings`,
`save_ints`, `save_floats`, `save_strings`). That module uses NumPy through
Python interop for file I/O only, like NuMojo's `load`/`save`.

`%%mojo <name> --expect-fail` documents an unsupported capability: the cell
passes only when the program fails to compile or run, and shows the error.

The Modular Mojo Jupyter kernel shipped with `mojo` 1.0.0 is not used: its
kernelspec contains an unexpanded `$PYTHON`, and once that is corrected its
REPL backend crashed on `print(1+1)` in our sandboxed environment while the
plain `mojo repl` worked. Compiled programs also match how Moxie is built and
tested.
