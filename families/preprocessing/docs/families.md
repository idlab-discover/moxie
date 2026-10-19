# Family repositories

Each sibling `moxie-<module>` repository starts from moxie tag `scaffold-v1.1`.
Public imports stay under `moxie.<module>`, matching the scikit-learn module
structure. These are interface scaffolds, not implemented estimators. Each
repository contains its own Pixi environment, Git history, family contracts,
compile checks, and API guide. `SCAFFOLD_BASE` records its exact foundation.

| Repository | Reference module | Family contract traits |
| --- | --- | --- |
| `moxie-linear_model` | `sklearn/linear_model` | `LinearRegressor`, `LinearClassifier`, `ProbabilisticLinearClassifier`, `OnlineLinearRegressor`, `OnlineLinearClassifier` |
| `moxie-svm` | `sklearn/svm` | `SupportVectorClassifier`, `ProbabilisticSVC`, `SupportVectorRegressor`, `OneClassSupportVector` |
| `moxie-neighbors` | `sklearn/neighbors` | `KNeighborsQuery`, `RadiusNeighborsQuery`, `KNeighborsClassifierAPI`, `KNeighborsRegressorAPI`, `RadiusNeighborsClassifierAPI`, `RadiusNeighborsRegressorAPI`, `NeighborDensity`, `NoveltyDetector` |
| `moxie-tree` | `sklearn/tree` | `TreeInspection`, `DecisionTreeClassifierAPI`, `DecisionTreeRegressorAPI` |
| `moxie-ensemble` | `sklearn/ensemble` | `EnsembleClassifier`, `EnsembleRegressor`, `ProbabilisticEnsemble`, `DecisionEnsemble`, `ForestInspection`, `StagedPredictor`, `IsolationDetector` |
| `moxie-naive_bayes` | `sklearn/naive_bayes.py` | `NaiveBayesClassifier` |
| `moxie-discriminant_analysis` | `sklearn/discriminant_analysis.py` | `DiscriminantClassifier`, `LinearDiscriminantTransformer` |
| `moxie-gaussian_process` | `sklearn/gaussian_process` | `GaussianProcessClassifierAPI`, `GaussianProcessRegressorAPI`, `GaussianProcessPosterior`, `GaussianUncertainty` |
| `moxie-neural_network` | `sklearn/neural_network` | `MLPClassifierAPI`, `MLPRegressorAPI`, `RBMTransformer` |
| `moxie-kernel_ridge` | `sklearn/kernel_ridge.py` | `KernelRidgeRegressor` |
| `moxie-dummy` | `sklearn/dummy.py` | `BaselineClassifier`, `BaselineRegressor` |
| `moxie-semi_supervised` | `sklearn/semi_supervised` | `SemiSupervisedClassifier` |
| Integrated: `src/moxie/multiclass.mojo` | `sklearn/multiclass.py` | `MulticlassClassifier`, `MulticlassDecision`, `MulticlassProbability` |
| `moxie-multioutput` | `sklearn/multioutput.py` | `MultiOutputClassifierAPI`, `MultiOutputRegressorAPI`, `MultiOutputProbability` |
| `moxie-calibration` | `sklearn/calibration.py` | `CalibratedClassifier` |
| `moxie-cluster` | `sklearn/cluster` | `Clusterer`, `PredictiveClusterer`, `ClusterDistanceTransformer`, `IncrementalClusterer`, `Biclusterer` |
| `moxie-mixture` | `sklearn/mixture` | `MixtureDensity`, `InformationCriteria` |
| `moxie-covariance` | `sklearn/covariance` | `CovarianceEstimator`, `RobustCovarianceDetector` |
| `moxie-decomposition` | `sklearn/decomposition` | `DecompositionTransformer`, `InvertibleDecomposition`, `IncrementalDecomposition` |
| `moxie-cross_decomposition` | `sklearn/cross_decomposition` | `CrossDecompositionTransformer`, `PLSPredictor` |
| `moxie-manifold` | `sklearn/manifold` | `EmbeddingEstimator`, `OutOfSampleEmbedding` |
| `moxie-preprocessing` | `sklearn/preprocessing` | `Preprocessor`, `InvertiblePreprocessor`, `OnlinePreprocessor`, `TargetPreprocessor` |
| `moxie-impute` | `sklearn/impute` | `Imputer` |
| `moxie-feature_selection` | `sklearn/feature_selection` | `FeatureSelector`, `SupervisedSelector`, `UnsupervisedSelector` |
| `moxie-feature_extraction` | `sklearn/feature_extraction` | `FeatureExtractor` |
| `moxie-kernel_approximation` | `sklearn/kernel_approximation.py` | `KernelFeatureMap` |
| `moxie-random_projection` | `sklearn/random_projection.py` | `RandomProjector` |
| `moxie-isotonic` | `sklearn/isotonic.py` | `IsotonicRegressorAPI` |
| `moxie-pipeline` | `sklearn/pipeline.py` | `PipelineEstimator`, `PredictivePipeline`, `TransformingPipeline` |
| `moxie-compose` | `sklearn/compose` | `ColumnComposer`, `TransformedTargetRegressorAPI` |
| `moxie-model_selection` | `sklearn/model_selection` | `SearchEstimator`, `PredictiveSearch`, `CrossValidator` |
| `moxie-frozen` | `sklearn/frozen` | `FrozenEstimatorAPI` |

## Porting scope

The table covers estimator-bearing families, including unsupervised learning,
feature preparation and estimator composition. `metrics`, `datasets`,
`inspection`, `utils`, `exceptions`, `callback`, configuration, and experimental
enablement are shared/supporting work, not duplicate algorithm-family repos.
Third-party `externals` and build internals are reference infrastructure rather
than Mojo algorithms to translate literally. The shared scaffold currently
implements only its documented validation helper; these supporting modules
remain future work.

Derive concrete algorithms inside the corresponding family, keeping the
reference class names and module locations. Family traits express shared
capabilities, not a frozen list of every constructor argument. Before starting
an algorithm, inspect its reference constructors, inherited methods, tests and
math documentation; add its exact typed contract and parity tests first.

## Local comparison sources

Family repositories keep a revision record and links to the local scikit-learn
source and user guide under `docs/private/reference/`. Specialized checkouts
belong under `docs/private/<algorithm>/sources/` with a URL, revision and license
record. These are local research inputs, never runtime dependencies or tracked
vendor copies. Deriving a repository with Git does not copy ignored resources;
prepare them again at the new algorithm location.
