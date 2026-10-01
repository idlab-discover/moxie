# Shared random engine

Own `moxie._random`. Read [RNG contract](docs/random.md) and [scope](docs/scope.md).
Use current `../assembly/tools/consume.py`; follow root live-source policy.
Dense results use NuMojo `NDArray`, `Shape`, `Item`. RNG words/index bookkeeping
remain algorithm-specific storage. Runtime: pure Mojo.
Run build, test, parity after changes; retain `--fp-mode=contract=off`.
