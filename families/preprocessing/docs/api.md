# Native preprocessing API

`moxie.preprocessing` exports the 29 public names of `sklearn.preprocessing`
(reference revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`). This page
lists the conventions shared by every estimator; module pages describe each
estimator's types, supported options and verified tolerances.

| Module page | Public names |
| --- | --- |
| [scalers](api/scalers.md) | StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, scale, minmax_scale, maxabs_scale, robust_scale |
| [simple](api/simple.md) | Normalizer, Binarizer, KernelCenterer, normalize, binarize, add_dummy_feature |
| [quantiles](api/quantiles.md) | QuantileTransformer, PowerTransformer, quantile_transform, power_transform |
| [labels](api/labels.md) | LabelEncoder, LabelBinarizer, MultiLabelBinarizer, label_binarize |
| [encoders](api/encoders.md) | OneHotEncoder, OrdinalEncoder, TargetEncoder |
| [polynomial](api/polynomial.md) | PolynomialFeatures, SplineTransformer |
| [function](api/function.md) | FunctionTransformer |
| [discretization](api/discretization.md) | KBinsDiscretizer |

```mojo
from numojo import NDArray, Shape, zeros
from moxie.base import clone
from moxie.preprocessing import StandardScaler

def main() raises:
    var X = zeros[DType.float64](Shape(4, 2))
    for k in range(8):
        X.unsafe_set(k, Float64(k * k))
    var scaler = StandardScaler()           # StandardScaler[DType.float64]
    scaler.fit(X)                           # mutates in place, returns nothing
    var Z = scaler.transform(X)             # new owned array
    scaler.transform_inplace(X)             # bounded-memory copy=False
    var fresh = clone(scaler)               # unfitted, same configuration
```

## Arrays and dtypes

Numeric estimators are parametrized by their floating dtype,
`Estimator[dtype: DType = DType.float64]`, and accept rank-2 NuMojo
`NDArray[dtype]` with samples in rows. Only Float32 and Float64 are accepted
(sklearn converts integer input to Float64; convert explicitly first). Input may
be any valid strided view, including negative strides and offsets. Inputs are
borrowed and never modified by copying methods; results are compact row-major
arrays owned by the caller. Learned attributes keep sklearn's dtype choices.

Label and categorical estimators use typed label/category containers described
on their module pages. Sparse matrices are unavailable by type: parameters that
select sparse output must be `False` (defaults that require sparse output raise a
clear `ValueError` at fit).

## Lifecycle

* Constructor keyword names, order, keyword-only boundaries and defaults match
  sklearn; `tests/api_surface.py` checks this against the pinned source, and
  `tests/api_adaptations.json` lists every documented exception.
* `fit`, `partial_fit` and `set_params` mutate in place and return nothing.
  `fit_transform` returns the transformed array.
* `get_params()` returns a typed `<Estimator>Params` record; `set_params(record)`
  replaces the configuration, and keyword overloads such as
  `set_params(with_mean=False)` update individual parameters. As in sklearn,
  set_params does not clear learned state; the next `fit` replaces it. Records
  name the `copy` parameter `copy_input` because `Copyable` reserves `copy()`.
* `moxie.clone(est)` returns an unfitted estimator with copied configuration.
* Learned attributes use sklearn names (`mean_`, `scale_`, `n_features_in_`).
  Before fit, arrays are empty and counts zero; methods that need learned state
  raise `NotFittedError`. Stateless transformers (Normalizer, Binarizer,
  FunctionTransformer) transform without fitting, as in sklearn.
* Errors are Mojo `Error` values whose text starts with the sklearn exception
  class (`ValueError:`, `NotFittedError:`, `TypeError:`) and, where practical,
  sklearn's message.

## Feature names

Arrays carry no column names. `fit_named(X, feature_names, y=None)` fits and
records `feature_names_in_`; ordinary `fit` clears it. `get_feature_names_out(
input_features=None)` follows sklearn: explicit names must match the fitted
names and count, otherwise fitted names or generated `x0, x1, ...` are used.

## Copy, in-place and memory

Copying methods allocate exactly the logical result plus O(features) state.
`copy=False` cannot alias a borrowed Mojo input, so it never mutates the
argument; instead estimators that sklearn can scale in place expose explicit
`transform_inplace(mut X)` / `inverse_transform_inplace(mut X)` methods that
require a writable array and allocate nothing. No method materializes
full-table masks, casts or residual temporaries (the reference does so in
several scalers); see each module page for its memory bound.

## Not supported

Python containers, pandas/polars output (`set_output` accepts only `None` or
`"default"`), metadata routing (`get_metadata_routing`, `set_*_request`),
sklearn tags, pickling and Python bindings are not provided. Python callables
become typed Mojo callables where an estimator needs them.

## Validation

`pixi run build` compiles the package, contract tests and examples;
`pixi run test` runs native contracts and the signature check; `pixi run parity`
runs `tests/parity/test_*.py`, which execute the pinned sklearn sources (an
overlay on an installed wheel's compiled helpers, see `tests/parity/overlay.py`)
and a compiled native driver on identical inputs. Parity covers the cases listed
on each module page and is not a claim of complete sklearn parity.
`pixi run memory` measures the peak RSS of every public scaler operation against
the byte bounds on [scalers.md](api/scalers.md).
