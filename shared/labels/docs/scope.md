# Label owner scope

Implemented contract: homogeneous String, Int64, Float32 and Float64 dense
rank-one/two targets; per-column sorted domains, integer inverse indices,
counts, first-occurrence order, lossless inverse decoding, existing-domain
lookup, owned snapshots, and dense classification-target identification.
See [labels.md](labels.md) for exact signatures and differences from sklearn.

Deferred: public preprocessing estimators, sparse/ragged/object/nullable targets,
row-string serialization specific to StratifiedShuffleSplit, warning routing,
Python bindings and downstream family adoption. No family is repinned by this
project and no missing capability returns a placeholder.

Numerical storage remains the accepted NuMojo NDArray. Standard-library sorting
and comparisons supply the core primitives; this project owns label semantics
and the native adaptation, not a generic sort or numerical backend.
