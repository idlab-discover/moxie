# GaussianNB dense scope

Implemented and integrated through `moxie.naive_bayes`: fit, weighted fit,
partial_fit, predict, all three probability methods, accuracy score,
configuration, unfitted clone and independent fitted copy.

The [API guide](api.md) defines array and List inputs, numeric/string target
domains and explicit Float32/Float64 statistics and probability dtypes.
Shared label preparation and normalization belong to the family and consume
Moxie's native labels.

Sparse inputs have no reference Gaussian branch and remain outside this CPU
dense implementation. Python bindings, metadata routing, named dataframe
columns and non-CPU array namespace dispatch remain outside the native scope.
This implementation does not claim exhaustive sklearn parity or performance.

Root build/test tasks compile the public facade, Gaussian contracts and both
isolated consumers. The Gaussian parity task compares typed array behavior to
the exact pinned sklearn module; the root differential suite checks List
compatibility and the other four estimators.
