# Foundation integration checks

Read [integration contract](docs/readiness.md). Fixtures test shared
`moxie._random`, `moxie._dense` and NuMojo storage; they implement no estimators.
Use current `../assembly/tools/consume.py` with explicit consumer paths.
Follow root live-source policy. Outputs/caches stay under each consumer's
`builds/` and `.pixi/`. Run build, test, parity after changes.
