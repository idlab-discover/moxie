# Dense-only scope

Supported in this family: pure Mojo dense numeric missing masks (Float32 and
Float64, NaN or finite markers), Boolean precomputed mask bookkeeping, retained
feature selection, indicator concatenation and owned iterative step records.
The family also exports the partial dense SimpleImputer and MissingIndicator
implementations described in [their API](../algorithms/simple/docs/api.md).
Dense Float32/Float64 KNNImputer reuses the neighbors query engine; see its
[API](../algorithms/knn/docs/api.md) for supported modes and differences.
IterativeImputer remains an explicit deferred declaration. Existing
Imputer trait contracts are retained.

Sparse input/output, CSR/CSC/COO storage, sparse masks and sparse concatenation
remain deferred.
Reference branches: sklearn/impute/_base.py SimpleImputer._sparse_fit/transform,
MissingIndicator._get_missing_features_info and _BaseImputer._concatenate_indicator.
KNN and iterative have dense-only reference input. The numerical dependency
supplies no sparse backend; this project neither imports one nor densifies input.
No public estimator export is removed on sparse grounds.

No supported estimator default needs sparse machinery. MissingIndicator uses
`sparse="auto"` on dense input or explicit `sparse=False`; `sparse=True` must
raise a clear unsupported-capability error. Other imputer defaults accept dense
input. Numeric helpers accept NDArray; the String specialization uses owned
StringMatrix text with explicit missing tags. Negative compilation verifies that a List cannot masquerade as matrix input.
Public API tests verify concrete imputation and explicit errors from deferred
constructors.

Executed acceptance commands: `pixi run build`, `pixi run test`.
These include two independent consumer build trees composing contributions to
one namespace, native validation/layout/ownership contracts, inherited contracts,
and rejection of drift/conflicting ownership/conflicting exports. Dense numeric
and Boolean indexing uses exact-value assertions; `pixi run parity` additionally
verifies floating, integer and string imputation branches against scikit-learn.
IntegerSimpleImputer supports signed/unsigned integer arrays with the documented
typed-result and range adaptations. StringSimpleImputer supports most-frequent
and constant strategies with the documented typed result and empty/inverse limits.
Standalone string missing indicators use `StringMissingIndicator`. Broader
heterogeneous categorical imputation and indicators remain unfinished, as does
IterativeImputer.
Passing these checks is not full dense estimator parity.
