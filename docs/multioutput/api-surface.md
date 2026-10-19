# Native multioutput API

Import the four estimators from `moxie.multioutput`. Supply a component that
implements `SupervisedEstimator` and `Predictor`; optional methods depend on
its capability traits.

```mojo
from moxie.multioutput import MultiOutputRegressor, ClassifierChain

# base_regressor consumes Float64 feature matrices and target vectors.
var model = MultiOutputRegressor(base_regressor^)
model.fit(X, Y)
var predictions = model.predict(X_test)
var score = model.score(X_test, Y_test)

# Classifier components consume Int64 target vectors by default.
var chain = ClassifierChain(base_classifier^, order=List[Int]([1, 0]))
chain.fit(X, labels)
var labels_predicted = chain.predict(X_test)
```

`X` has shape `(samples, features)` and `Y` has shape `(samples, outputs)`,
including a single output. `predict` returns `(samples, outputs)`.
Use explicit type parameters `[Component, dtype, label]` for other numeric
feature/target dtypes. Regression targets default to the feature dtype;
classification targets default to Int64.

`fit`, `partial_fit` and `set_params` mutate in place and return a mutable borrow of the wrapper, supporting
`model.fit(X, Y).predict(X)`. `get_params(deep=True)`
returns a typed record containing the component's parameter record. Edit that
record and pass it to `set_params`. `clone(model)` recursively clones component
configuration into an unfitted wrapper. A successful ordinary fit replaces all
fitted components; a failed ordinary fit leaves the prior model intact.

Independent estimators accept `n_jobs=None` or `1`. Their `partial_fit` exists
only when the component has the appropriate incremental trait. A classifier's
first incremental fit requires `classes=List[NDArray[label]]`, one complete
class vector per output. Subsequent calls retain each fitted component.
`fit(X, Y, sample_weight=weights)` requires a component with weighted-fit support.

Chain constructors accept `order=None`, `random_state=None`,
`random_order=False`, and `verbose=False`. `ClassifierChain` also accepts a
`chain_method` string, defaulting to `"predict"`. Random order is requested with
`random_order=True`; it is mutually exclusive with an explicit permutation.
`verbose=True` reports completed links during ordinary fit. Native CV uses
`fit(X, Y, cv=provider)` with the `ChainResponses` protocol. Its returned vector
must contain one out-of-fold response per original sample; the final link does
not request a CV response.

Regression chains always use `predict` responses. The native regression
parameter record also contains `chain_method` for compatibility with the shared
chain configuration, but both ordinary and CV fitting reject values other than
`"predict"`. Scikit-learn exposes this parameter only on classifier chains.

Learned state includes `estimators_`, `n_features_in_`, classifier `classes_`,
and chain `order_` and `chain_method_`. `n_features_in_` is an `Optional[Int]`;
`None` denotes an unfitted model. Chain class vectors are in chain order.
Independent class vectors are in original output order.

Classifier components must use the sorted unique training labels as their class
vocabulary and probability-column order. For incremental fitting, they must use
the complete class vectors supplied to the wrapper. The current component
traits do not expose fitted class metadata. Frozen or fixed-vocabulary
classifiers whose classes differ from these vectors are unsupported, as are
components with a different probability-column order.

`MultiOutputClassifier.score` computes exact row matches and rejects sample
weights. Classifier chains accept weighted subset accuracy. Regression scores
use uniform-average R2, force finite scores for constant targets, and return NaN
with fewer than two samples. Native errors use descriptive Mojo `Error` values.

See [scope](scope.md) for unsupported inputs and configuration options.

`IndependentParams` and `ChainParams` are public typed records in
`moxie.multioutput`. Capability contracts remain available there and in
`moxie._multioutput_contracts`; `ChainResponses` remains available there and in
`moxie._multioutput`.

The weighted `fit` overload is available only for components implementing
`WeightedSupervisedEstimator`. Moxie's `DummyRegressor` is directly compatible
with independent and chained regression. DummyClassifier and current Naive Bayes
classifiers use List target contracts, so they are not direct NDArray component
substitutes. No conversion adapter is supplied by these wrappers.

Wrappers are `Copyable` when their component is `Copyable`. A fitted copy owns
separate component state under that component's copy contract; cloning remains
unfitted. Copy support does not impose a requirement on move-only components.
