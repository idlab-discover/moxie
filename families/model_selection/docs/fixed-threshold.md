# FixedThresholdClassifier

```mojo
from moxie.frozen import FrozenEstimator
from moxie.linear_model import LogisticRegression
from moxie.model_selection import FixedThresholdClassifier

var classifier = LogisticRegression()
_ = classifier.fit(X_train, y_train)
var model = FixedThresholdClassifier(FrozenEstimator(classifier^), threshold=0.9)
var labels = model.predict(X_test)
```

The wrapper accepts an owned classifier and can predict immediately if it is
fitted. An explicit fit clones the constructor estimator and stores the result
in `estimator_`. The constructor estimator keeps its state. A frozen estimator's
clone preserves its shared fitted model, so fitting the wrapper with new binary
targets leaves the frozen model fitted on the original training data.

`threshold=None` means automatic. `response_method="auto"` selects probability
when the estimator provides it, then decision function. The automatic cutoff is
0.5 for probability and 0 for decision. Scores equal to the cutoff predict the
positive label. By default the second fitted class is positive. To select the
first class, pass its typed label, such as `pos_label=Int64(-3)` for the default
logistic regression label dtype. This selects the first probability column or
reverses the sign of the binary decision score.

`classes_()` reads the ordered fitted labels and `get_n_features_in()` forwards
the estimator's learned width when that trait is available. Probability,
log probability and decision methods forward conditionally and retain the
estimator's output dtype and shape. `score` computes accuracy on the
thresholded predictions. The wrapper accepts typed `sample_weight` for fit
when the estimator implements weighted fit and accepts weights for accuracy.
Use the typed `FixedThresholdParams` from `get_params()` to clear an optional
setting. Named updates affect supplied parameters, and `set_params(estimator=...)`
replaces the constructor estimator. Like scikit-learn, the wrapper uses an
existing `estimator_` until another fit succeeds.

The dense class-order contract is `NumericClassifierClasses` in `moxie.base`.
It supports Float32 and Float64 features and responses, with Int64, Float32 or
Float64 array targets or `List[Int]` targets. `LogisticRegression` with those
target dtypes and `LinearDiscriminantAnalysis` with `Int` labels implement it.
The frozen wrapper forwards the contract. Clone preserves the fitted state of
a frozen constructor estimator and clones an ordinary constructor estimator
according to its usual unfitted-clone rule.

`tests/frozen/test_fixed_threshold.mojo` checks fitted state, clone and refit,
public method forwarding, invalid targets, thresholds and response selection.
`tests/frozen/compare_fixed_threshold.py` compares predictions and scores
exactly with scikit-learn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. The frozen gallery compares
the complete test-set predictions at a cutoff of 0.9 and the delegated
probabilities.
