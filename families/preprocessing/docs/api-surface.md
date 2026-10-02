# Public API surface

`moxie.preprocessing` exports the 29 public names of `sklearn.preprocessing`
at the recorded reference revision, each with a native implementation for
dense input. Constructor and function parameter names, order, keyword-only
boundaries and defaults follow the local scikit-learn checkout; language-driven
exceptions are listed in `tests/api_adaptations.json` and documented on each
[module page](api.md).

`pixi run test` compiles the package and checks:

- `tests/api_surface.py`: the 29 export names and their signatures against the
  pinned scikit-learn source;
- `tests/deferred_surface.py --require-complete`: no public entry point still
  raises the historical `NotImplementedError: ... is an API scaffold` error.

These checks establish declaration coverage, not numerical parity. Parity is
measured separately by `pixi run parity` (see [api.md](api.md)). Metadata
routing, Python reflection and sparse matrices remain unsupported, as stated
in [api.md](api.md) and [scope.md](scope.md).
