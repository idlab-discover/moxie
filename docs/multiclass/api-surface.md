# Public multiclass API

The three public estimators in the pinned `sklearn.multiclass.__all__` are
implemented for dense numeric or string single-label targets; OvR additionally supports
dense multilabel indicator targets:

- [OneVsRestClassifier](one_vs_rest.md)
- [OneVsOneClassifier](one_vs_one.md)
- [OutputCodeClassifier](output_code.md)

Their guides describe constructor defaults, learned attributes, optional
capabilities, validation evidence and remaining limitations. Public imports
remain under `moxie.multiclass`. Typed parameter records are exported alongside
the estimators; fluent mutation and dynamic Python facilities follow the
[shared interface adaptations](../interfaces.md).

All three wrappers accept classifier and regressor components. OutputCodeClassifier
requires `DecisionFunction` or `ProbabilityPredictor` even for regressors, while
decoding their `predict` output. See each guide for the typed component contracts.
OvR and OvO provide numeric- and string-label `partial_fit` when the component
conforms to `IncrementalClassifier`; the OvO guide documents the string component
target adaptation. Select `kind=StringLabels` with shared `StringTarget` inputs
for string labels.

No public estimator remains a raising scaffold. This does not establish full
family parity: sparse inputs, metadata routing and other guide-listed
capabilities remain unfinished.

Dense precomputed inputs use the family-local `PairwiseInput` capability.
OvO slices each batch-training kernel by both pair axes and retains indices
for prediction; OvR passes through the full matrix and propagates the capability.
OutputCode passes through matrices without propagating it, matching sklearn.
See the estimator guides for incremental behavior and validation boundaries.
