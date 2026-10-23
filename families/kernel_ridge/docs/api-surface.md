# KernelRidge parity surface

The reference is the local scikit-learn checkout at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
The active implementation is `src/moxie/kernel_ridge.mojo` and its private
`src/moxie/_kernel_ridge/` modules. Build and test from the Moxie root; see [commands](../README.md).
See [family.md](family.md) for the public API, language adaptations and
numerical tolerances.

| Surface | Current support |
| --- | --- |
| Estimator | `KernelRidge`, with scikit-learn constructor names/defaults |
| Lifecycle | `fit`, `predict`, `score`, `get_params`, `set_params`, unfitted clone and fitted copy |
| Learned attributes | `dual_coef_`, `X_fit_`, `n_features_in_`, optional `feature_names_in_` |
| Dense data | Mixed Float32/Float64 and Int32/Int64, strided matrices, vector/multioutput targets |
| Dtypes | Runtime Float32/Float64 coefficient/prediction variants; kernel-, weight- and alpha-dependent promotion |
| Kernels | All named kernels, precomputed matrices and typed native callables |
| Weights | Scalar/vector fit and score weights; target-specific alpha |
| Sparse feature inputs | Not implemented; reference accepts CSR and CSC |
| Feature names | Explicit `feature_names` keyword on fit/predict/score; exact ordered validation |
| Metadata requests | `get_metadata_routing`, `set_fit_request`, `set_score_request`; native sample-weight selection/rejection and aliases |
| Parameters | `get_params`/mapping updates plus optional typed records; direct optional resets |
| Composition | `clone`, final step of `moxie.pipeline` pipelines, `cross_val_score`/`learning_curve`/`GridSearchCV` through typed model_selection adapters (`tests/test_composition.mojo`, `notebooks/kernel_ridge/`) |
| Python integration | Dataframe adapters, Python callbacks, unrestricted kwargs and global sklearn routing remain outside the native API |

## Deferred: sparse feature inputs

A remaining estimator capability is CSR/CSC input for fit and
prediction. The reference explicitly accepts these in `KernelRidge.fit` and
`KernelRidge.predict`; `sklearn/tests/test_kernel_ridge.py` exercises both
formats in `test_kernel_ridge_sparse`.

Start with the linear kernel and dense targets, retaining sparse training data
in `X_fit_` and using a dense Gram matrix with the existing solver. Define the
shared sparse storage contract in Moxie before introducing a reusable container;
keep KernelRidge-specific dispatch and kernel operations in this family.
Then extend the other kernels according to their actual sparse support in the
pinned `sklearn/metrics/pairwise.py`, including explicit unsupported cases.

Acceptance should compare CSR and CSC with the reference across both dtypes,
weighted and multioutput fitting, dense/sparse prediction combinations,
repeated fits, cloning, invalid sparse structure, duplicate/unsorted entries,
explicit zeros and empty rows. Avoid densifying the feature matrix: the dense
Gram allocation is intrinsic to this solver, not a reason to discard sparse
feature storage. Sparse targets and sparse output are separate capabilities.
