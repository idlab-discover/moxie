# Foundation readiness

Reuse the read-only assembler at ../assembly/tools/consume.py with explicit
consumer paths. Pin moxie._random.RandomState from ../random, moxie._dense
sum_axis2d and row_squared_norms from ../numerics, and its accepted numojo
NDArray, Shape and Item. Read docs/readiness.md before extending the gate.
Fixtures demonstrate integration only; estimator ownership stays with families.
All generated outputs and caches belong under each consumer’s ignored builds/
and .pixi/. Dependency changes require an explicit lock update, never a build.
