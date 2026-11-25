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

Independent estimators use `n_jobs` for `fit` and `partial_fit`. `None` and `1`
use the calling thread. Positive values cap the number of MAX CPU workers,
and negative values count back from the logical core count, with `-1` using
all cores. The worker count stays between one and the output count. `0`
raises. Components trained concurrently must support independent fits with
shared, read-only inputs. Inference traverses the components serially.

`partial_fit` exists only when the component has the appropriate incremental
trait. A classifier's first incremental fit requires
`classes=List[NDArray[label]]`, one complete class vector per output.
A successful first incremental fit publishes all components together.
Subsequent calls update components in place, so a failing call can leave
some components updated. When multiple parallel tasks fail, the wrapper
reports the error from the lowest target-column index.
`fit(X, Y, sample_weight=weights)` requires a component with weighted-fit support.

Chain constructors accept `order=None`, `random_state=None`,
`random_order=False`, `verbose=False`, and `cv=None`. `ClassifierChain` also
accepts a `chain_method` string, defaulting to `"predict"`, or a preference
list such as `List[String](["predict_proba", "decision_function", "predict"])`.
At fit time the chain selects the first response supported by the component's
traits and records its name in `chain_method_`. An empty list or a list with
no supported method raises a Mojo `Error`. Component-specific errors from the
selected response propagate when that method is called.

Classifier chain parameters store this setting as `ChainMethod`, a
`Variant[String, List[String]]` from `moxie.multioutput`. Use the estimator's
`Params` type or infer the record from `get_params()`. For a string setting,
`params.chain_method[String]` reads the configured name. Both strings and
lists can be assigned to `params.chain_method`. Cloning and parameter updates
copy the configuration. Fit resolves it for out-of-fold responses and
inference uses the fitted name.

Use `random_order=True` to request a random permutation in the absence of an
explicit order.
`verbose=True` prints sklearn's `[Chain] ... (i of n) Processing order k, total=...`
line after each link is fitted. Set `cv=3`, `cv=KFold(3)`, or
`cv=StratifiedKFold(3)` on either chain constructor. Import the splitters from
`moxie.model_selection`. To reuse explicit train/test indices, pass
`cv=ChainFolds(splits)` with a `List[IndexSplit]`, or materialize a splitter's
cursor with `cv=ChainFolds(splitter.split(n_samples))`. Import `ChainFolds`
from `moxie.multioutput` and `IndexSplit` from `moxie.model_selection`.
`ChainFolds` owns its indices and reuses the same splits for every link that
needs out-of-fold responses. Those responses require the test indices to
cover each sample exactly once.

For folds that depend on the current link, pass a native `ChainSplitter`
to either constructor as `cv=splitter`. Implement `split(X, y)` with the
chain's feature matrix and target vector types, and return a fresh
`moxie.model_selection.SplitCursor`. The splitter sees the augmented
features and original target column for each link preceding the last.
The chain checks that the test indices form a partition and handles fitting
and scattering the held-out responses.

Mojo infers the custom splitter type when calling
`ClassifierChain(component, cv=splitter)`. When specifying types explicitly,
use `[Component, dtype, label, Splitter]`. The resulting parameter record
stores that concrete splitter type, so `set_params` can update its
configuration. Built-in CV options use the default `ChainCV` variant.

The `ChainParams.cv` value is carried by `get_params`,
`set_params`, and `clone`. Native CV uses the model-selection target dtypes
Int64, Float32 and Float64. Integer CV uses unshuffled KFold, with stratification
for a classifier component and binary or multiclass targets. A shuffled native
splitter requires an integer `random_state`.

A built-in CV option passed to `fit` overrides the configured value for
that call. Use `fit(X, Y, cv=None)` for true preceding targets or
`fit(X, Y, cv=provider)` with a custom `ChainResponses` provider.
The provider returns one out-of-fold
response per original sample. Every link fits on the full data, and only links
preceding the final one request CV responses. When configured CV is combined
with `sample_weight`, the full-data links receive the weights. The out-of-fold
fits use unweighted data, following scikit-learn's chain behavior.

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
the complete class vectors supplied to the wrapper. The wrappers derive
these vocabularies from the supplied labels. Frozen or fixed-vocabulary
classifiers whose classes differ from these vectors are unsupported, as are
components with a different probability-column order.

`MultiOutputClassifier.score` computes exact row matches and rejects sample
weights. Classifier chains accept weighted subset accuracy. Regression scores
use uniform-average R2, force finite scores for constant targets, and return NaN
with fewer than two samples. Native errors use descriptive Mojo `Error` values.

See [scope](scope.md) for unsupported inputs and configuration options.

`IndependentParams` and `ChainParams` are public typed records in
`moxie.multioutput`. `ChainCV` is the native splitter configuration type.
Capability contracts are also available in `moxie._multioutput_contracts`.
The `ChainResponses` protocol can also be imported from `moxie._multioutput`.

The weighted `fit` overload is available only for components implementing
`WeightedSupervisedEstimator`. Tested production components are `DummyRegressor`,
`Ridge` and `RandomForestRegressor` (regression, independent and chained) and
`RidgeClassifier`, `LogisticRegression` and `RandomForestClassifier`
(classification; forests use `[Forest, DType.float32, DType.float64]`).
`LogisticRegression()` supplies the default Float64 features and Int64 labels
and supports both independent probabilities and chain probability responses.
The wrappers also accept `DecisionTreeClassifier` and
`KNeighborsClassifier` with numeric NDArray targets. Each fitted link has one
target, so the wrapper takes the sole matrix from the component's probability
list. Use Float32 features for trees and the chosen component dtype for
neighbors. Both classifier wrappers accept single-target components whose targets
and predictions use numeric lists. The default Int64 wrapper labels work with
`GaussianNB()` and `LinearDiscriminantAnalysis()`, which use `List[Int]`.
A component using `List[Float64]` labels requires the explicit wrapper label
dtype `DType.float64`. Fits, native CV folds, and incremental class declarations
use the component's native representation. Wrapper targets, predictions and
`classes_` remain NDArrays. The wrappers convert Float64 NDArray weights to the
`List[Float64]` that GaussianNB consumes.

`DummyClassifier()` uses tagged `Label` values internally. Either wrapper
accepts it with Int64, Float32 or Float64 targets. With a constant strategy,
use an integer `Label` for Int64 targets or a floating-point `Label` for
floating-point targets. `SVC[Int]()` works with the default Int64 wrapper
labels and Float64 features.

Independent wrappers fit and expose their components in target-column order.
`MultiOutputClassifier.partial_fit` with `classes` also records `classes_`,
which scikit-learn sets only in `fit`.

Wrappers are `Copyable` when their component is `Copyable`. A fitted copy owns
separate component state under that component's copy contract; cloning remains
unfitted. Copy support does not impose a requirement on move-only components.
