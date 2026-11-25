"""Use the public metrics through model_selection's cross-validation scorer seam."""
from std.testing import assert_false

from moxie.metrics import accuracy_score, balanced_accuracy_score
from moxie.model_selection import CVAdapter, ErrorPolicy, IndexArray, IndexSplit
from moxie.model_selection import KFold, TypedScorer, cross_val_score, select_rows
from moxie.naive_bayes import GaussianNB, NumericDomain
from numojo import Item, NDArray, Shape

comptime Dense = NDArray[DType.float64]
comptime Labels = NDArray[DType.int64]
comptime Classifier = GaussianNB[NumericDomain[DType.int64]]


@fieldwise_init
struct AgreementScorer(TypedScorer):
    var balanced: Bool

    comptime Estimator = Classifier
    comptime Input = Dense
    comptime Target = Labels
    comptime Metadata = Optional[Dense]

    def score(
        self, estimator: Classifier, X: Dense, y: Labels, metadata: Self.Metadata
    ) raises -> Float64:
        var prediction = estimator.predict(X)
        if self.balanced:
            return balanced_accuracy_score(y, prediction, sample_weight=metadata)
        return accuracy_score(y, prediction, sample_weight=metadata)


@fieldwise_init
struct Adapter(CVAdapter):
    comptime Scorer = AgreementScorer

    def n_samples(self, X: Dense, y: Labels) raises -> Int:
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise Error("ValueError: incompatible feature and target shapes")
        return X.shape[0]

    def input_rows(self, X: Dense, rows: IndexArray, train: IndexArray) raises -> Dense:
        return select_rows(X, rows)

    def target_rows(self, y: Labels, rows: IndexArray) raises -> Labels:
        return select_rows(y, rows)

    def metadata_rows(self, metadata: Optional[Dense], rows: IndexArray) raises -> Optional[Dense]:
        if metadata:
            return select_rows(metadata.value(), rows)
        return None

    def fit(self, mut estimator: Classifier, X: Dense, y: Labels, metadata: Optional[Dense]) raises:
        _ = estimator.fit(X, y)


def main() raises:
    var features: List[Float64] = [0, 1, 7, 8, 2, 3, 6, 9, 0.5, 2.5, 7.5, 9.5]
    var labels = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    var X = Dense(Shape(len(labels), 1))
    var y = Labels(Shape(len(labels)))
    var weights = Dense(Shape(len(labels)))
    for i in range(len(labels)):
        X[Item(i, 0)] = features[i]
        y[Item(i)] = Int64(labels[i])
        weights[Item(i)] = Float64(i + 1)

    var model = Classifier()
    var splits = List[IndexSplit]()
    var cursor = KFold(3).split(len(labels))
    while True:
        var split = cursor.next()
        if not split:
            break
        splits.append(split.value().copy())

    var scores = cross_val_score[Adapter](
        model, X, y, Adapter(), AgreementScorer(balanced=False), splits, None, weights.view(),
        error_score=ErrorPolicy(True, 0),
    )
    assert_false(model.__sklearn_is_fitted__())
    print("cv", scores[Item(0)], scores[Item(1)], scores[Item(2)])

    var balanced_scores = cross_val_score[Adapter](
        model, X, y, Adapter(), AgreementScorer(balanced=True), splits, None, weights^,
        error_score=ErrorPolicy(True, 0),
    )
    assert_false(model.__sklearn_is_fitted__())
    print("cv_balanced", balanced_scores[Item(0)], balanced_scores[Item(1)], balanced_scores[Item(2)])
