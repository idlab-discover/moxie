# preprocessing family interfaces

This repository declares capabilities, not algorithm implementations.
It uses scikit-learn revision `4358542a1bd39af2e7e1cbace8b9fed067473c5b`.
Read [the shared contract](interfaces.md) for typed parameter records,
in-place fit, associated array types, and other Mojo-specific differences.

## Public contracts

| Trait | Shared capabilities | Additional required methods |
| --- | --- | --- |
| `Preprocessor` | `UnsupervisedEstimator, TransformerMixin, FitTransformer` | Inherited only |
| `InvertiblePreprocessor` | `Preprocessor, InverseTransformer` | Inherited only |
| `OnlinePreprocessor` | `IncrementalTransformer` | Inherited only |
| `TargetPreprocessor` | `SupervisedEstimator, TransformerMixin` | Inherited only |

## Behavioral distinctions

Feature scalers/encoders differ from target-only encoders. TargetEncoder fit_transform uses cross-fitting and is not equivalent to fit().transform(). Feature names, sparse output and unknown-category rules belong to concrete estimators.

Associated result types preserve estimator-specific shapes. A return value
that changes structure with a Boolean option needs a suitable tagged/optional
result type in the concrete implementation; an integer probe in the tests is
only a signature check. Constructors, fit metadata, learned fields and
estimator-specific overloads must be specified before implementing an algorithm.

## Reference sources

- Source: `../scikit-learn/sklearn/preprocessing`.
- Guide and mathematical references: `../scikit-learn/doc/modules/preprocessing.rst`.
- Shared contracts: `../scikit-learn/sklearn/base.py` and `doc/developers/develop.rst`.
- Tests: the corresponding `tests/` directory or `sklearn/tests/test_<module>.py`
  in the local reference checkout.

The source inventory, exact reference revision, license link, and local source
links live in `docs/private/reference/`. Follow citations in the user guide and
class docstrings when implementing a concrete algorithm. Place specialized
sources and papers under `docs/private/<algorithm>/`; record revisions there.

## Reference classes to investigate

`OneHotEncoder`, `OrdinalEncoder`, `MinMaxScaler`, `StandardScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `Binarizer`, `KernelCenterer`, `QuantileTransformer`, `PowerTransformer`, `KBinsDiscretizer`, `FunctionTransformer`, `LabelEncoder`, `LabelBinarizer`, `MultiLabelBinarizer`, `PolynomialFeatures`, `SplineTransformer`, `TargetEncoder`.

This roster is extracted from source class definitions, not a promise that all
names are public exports. Inspect `__all__`, module exports and inherited
methods before defining each concrete public estimator.

## Dense plumbing

Read [plumbing.md](plumbing.md) for shared interfaces, operation restrictions,
assembly, extension ownership and all nine child sessions. [scope.md](scope.md)
defines the authorized sparse exclusions and the checks executed by this scaffold.
