# Public manifold API

The ten scikit-learn manifold exports have native dense implementations. See
[ClassicalMDS and trustworthiness](algorithms.md), [dense TSNE](tsne.md),
[dense Isomap](isomap.md), [dense MDS and SMACOF](mds.md), [dense LLE](lle.md),
and [dense spectral embedding](spectral.md) for supported settings and native
signatures. The [scope](scope.md) records sparse exclusions and explicit
settings required for dense operations.

Estimators use typed configuration and native arrays. Dedicated result records
represent functions that return additional values. Array dtype is a compile-time
parameter. Clone copies configuration into an unfitted estimator, and learners
expose attributes with their reference names. Python reflection and dynamic
metadata routing remain binding work.

The family build assembles current local modules and explicit public exports.
Framework and numerical sources come from their current local checkouts.
Use the checks and pinned oracle commands in each algorithm's documentation.
