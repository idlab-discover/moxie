# Dense ensemble scope

Supported: dense family records and extension traits, snapshot ownership,
resolved sampling/count/name/class-alignment validation, estimator seed draws,
warm-start/OOB configuration validation, and the dense forest engine
(bootstrap/bagging sampling, class weighting, serial member building, averaging,
OOB scores, importances; see forest_engine.md). Tests exercise both Float32/Float64
snapshots, shared dense reductions/RNG, clone lifecycle through isolated consumer
probes, and namespace collision and missing-source rejection. Public
RandomForestClassifier/Regressor and ExtraTreesClassifier/Regressor now
use this engine; see [forests.md](forests.md) for supported types, methods and
limits. Other public estimator names remain declarations that raise. Boosting,
bagging, voting, stacking, isolation and random embedding are not implemented.

Deferred sparse branches: all CSR/CSC/COO inputs, conversions, sparse passthrough,
and tree/forest decision_path. Reference: sklearn/ensemble/_forest.py decision_path,
_bagging.py input validation, _stacking.py concatenation. RandomTreesEmbedding's
sparse_output=True default is unsupported in dense work: its future child must
require sparse_output=False explicitly and test dense one-hot results. No sparse
input is automatically densified. Histogram boosting already requires dense input.
Inherited ForestInspection remains a legacy signature, not a sparse implementation;
new consumers use DenseForestInspection. Tree dependency has the same restriction.

Executed dense checks: `pixi run ensemble-build` and `pixi run ensemble-test` (both passed). Test includes an expected
compile failure for decision_path on the dense trait, raising engine probes,
invalid sampling/count/class/name/OOB settings, and two independently assembled
consumers. `pixi run ensemble-parity` checks the forest engine and sampling bit for bit against the
reference revision (forest_engine.md lists the one known upstream tree divergence).
Performance and parallel building are not claimed.
Authorized sparse deferrals do not remove any dense engine from its future session.
