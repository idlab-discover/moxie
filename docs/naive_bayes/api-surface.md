# Dense scikit-learn API coverage

Reference: local scikit-learn `4358542a1bd39af2e7e1cbace8b9fed067473c5b`
(`1.10.dev0`), `sklearn/naive_bayes.py::__all__`. All five public estimators
exist. Sparse inputs are deferred; the gaps below concern dense inputs.

| Estimator | Dense feature inputs | Targets | Per-feature configuration | Implementation |
| --- | --- | --- | --- | --- |
| GaussianNB | Float64 Lists; supported dense NDArray | List[Int], numeric NDArray, StringTarget | Not applicable | `_nb_gaussian` |
| MultinomialNB | Float64 Lists; supported dense NDArray | List[Int], numeric NDArray, StringTarget | `alpha_values` | `_nb_counts` |
| ComplementNB | Float64 Lists; supported dense NDArray | List[Int], numeric NDArray, StringTarget | `alpha_values` | `_nb_counts` |
| BernoulliNB | Float64 Lists; supported dense NDArray | List[Int], numeric NDArray, StringTarget | `alpha_values` | `_nb_bernoulli` |
| CategoricalNB | Float64 Lists; supported dense NDArray | List[Int], numeric NDArray, StringTarget | Scalar alpha; `min_categories_values` | `_nb_categorical` |

Every estimator exposes the reference constructor defaults and learned attribute
names, fit, weighted fit, partial_fit, predict, predict_proba,
predict_log_proba, predict_joint_log_proba, score, get_params and set_params.
Clone returns an unfitted model; copy preserves independently owned fitted state.
These methods are implemented, but their presence is not full behavioral parity.

BernoulliNB vector alpha uses correct per-feature smoothing; this deliberately
fixes the pinned reference's denominator broadcasting bug. See
[the documented behavior](family.md) and the scalar-fit composition oracle in
`tests/naive_bayes/check_sklearn.py`. Sparse input remains deferred. All five classifiers
support numeric and string target domains.

Supported feature dtypes are `bool`, signed/unsigned 8/16/32/64-bit integers,
`float32` and `float64`. `fit` and `partial_fit` accept Float64 scalar weights
(broadcast per row) as well as Float64 Lists; `score` weights remain Lists.

Every classifier uses a compile-time label domain; for example, `GaussianNB()` preserves
List[Int] targets; `GaussianNB[NumericDomain[DType.float64]]()` accepts integral
Float64 labels, and `GaussianNB[StringDomain]()` accepts StringTarget.
Use `GaussianNB[]` when annotating the default concrete type. Gaussian accepts an explicit
`dtype=DType.float32` parameter for Float32 arithmetic, learned NDArrays and
probabilities; its default remains Float64. Other classifiers retain Float64
statistics. See
[Gaussian API](gaussian/api.md) for shapes and examples.

Other explicit differences remain in [family.md](family.md): typed parameter
records, fluent mutable-reference returns, descriptive Mojo errors, stricter
negative-weight/prior validation and default Gaussian arithmetic dtype. See
[behavior classifications](family.md#behavioral-compatibility-decisions). Named-data
feature_names_in_, metadata routing, Python reflection/bindings and device array
namespace dispatch are outside the native contract. Gaussian is CPU dense only.
Multioutput targets are unsupported by the reference classifiers as well.

`pixi run test` checks native contracts and composed consumers.
`tests/naive_bayes/check_sklearn.py` checks array, List, weighted, incremental and typed-label
family differential fixtures;
`tests/naive_bayes/gaussian/parity.py` adds array/typed-label Gaussian fixtures.
Neither a finite fixture set nor compilation establishes exhaustive parity.
