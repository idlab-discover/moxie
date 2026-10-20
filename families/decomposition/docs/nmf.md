# Dense NMF and MiniBatchNMF

`from moxie.decomposition import NMF, MiniBatchNMF, non_negative_factorization`
factorizes dense non-negative Float32/Float64 input as `X ~ W H`, following
local scikit-learn `sklearn/decomposition/_nmf.py` and `_cdnmf_fast.pyx` at
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`.

```mojo
var nmf = NMF(n_components=2, init=Optional[String]("nndsvda"), random_state=0)
var W = nmf.fit_transform(X)        # (n_samples, 2), H = nmf.components_.value()
```

See [the example](../examples/nmf.mojo).

## API

Constructor names/defaults follow sklearn. Tagged alternatives use Variants:
`n_components: NMFComponents` (int, `"auto"` default, or `NoneType()`),
`beta_loss: BetaLoss` (`"frobenius"`, `"kullback-leibler"`, `"itakura-saito"`
or a Float64), `alpha_H: AlphaH` (Float64 or `"same"`); `init` is
`Optional[String]` (`random`, `nndsvd`, `nndsvda`, `nndsvdar`, `custom`).
NMF adds `solver` (`"cd"` Frobenius only, `"mu"` any beta) and `shuffle`;
MiniBatchNMF adds `batch_size=1024`, `max_no_improvement=10`,
`forget_factor=0.7`, `fresh_restarts=False`, `fresh_restarts_max_iter=30`,
`transform_max_iter=None`.
Learned: `components_`, `n_components_`, `reconstruction_err_`, `n_iter_`,
`n_features_in_` (plus `n_steps_` for MiniBatchNMF).
Methods: `fit`, `fit_transform(X)` and `fit_transform(X, W, H)` (custom init),
`transform` (solve W with H fixed), `inverse_transform`, MiniBatchNMF
`partial_fit(X)`/`partial_fit(X, W, H)`, `get_feature_names_out`, typed
`get_params`/`set_params` (`NMFParams`, `MiniBatchNMFParams`), clone.
`non_negative_factorization` returns an `NMFResult` record `(W, H, n_iter)`
and supports `update_H=False`.

Reference behaviors reproduced: per-call re-seeding from `random_state` for
initialization, NNDSVD SVD and coordinate-descent shuffles; data-shape scaled
L1/L2 regularization; float32-eps guards in the divergences; the error check
every 10 MU iterations; mini-batch EWA cost / no-improvement / H-change
convergence and in-place W row updates; and, for KL with a fixed H
(`transform`, `update_H=False`), the reference's cached H row sums that its
in-place `denominator += l1_reg_W` grows each iteration. Not reproduced:
warnings (check `n_iter_ == max_iter`), `verbose` printing, sparse input.

## Validation

`pixi run parity` runs 156 cases against the pinned source; its Cython kernel
`_cdnmf_fast.pyx` is compiled from the pinned file for the comparison.
Coverage: tall/wide/square shapes with exact zeros, all inits, cd with
shuffle and L1/L2 mixes, truncated `max_iter`, MU with Frobenius, KL,
IS, beta 0.5/1.5/3, custom W/H, `update_H=False` for both solvers,
MiniBatchNMF with several batch sizes, KL/IS, fresh restarts, regularization
and partial_fit, in Float64 and Float32. W, H, transforms and errors are
compared directly with tolerance 1e-7 (Float64) and 3e-3 (Float32), scaled by
the data magnitude; Float64 iteration/step counts must match exactly, Float32
may differ by up to 10 (rounding-level stopping decisions).
