# Assembly scope

Current behavior: [assembly contract](assembly.md). Live local inputs, validated
paths/ownership/exports, collision checks, baseline deduplication, isolated
consumer namespaces. No source revision/hash gates.

Native fixtures exercise shared dense storage and estimator contracts. Fixtures
are test doubles, not implemented estimators. Assembly adds no numerical kernels,
sparse storage or estimator behavior. Numerical exclusions belong to their owners.

Run `pixi run build`, `pixi run test`, `pixi run parity`.
