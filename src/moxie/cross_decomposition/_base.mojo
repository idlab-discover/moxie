"""Capabilities shared by the dense cross-decomposition estimators."""
from moxie.base import RegressorMixin, SupervisedEstimator, TransformerMixin
from moxie.compose._table import ColumnTable
from moxie.cross_decomposition._contracts import Matrix, PairedScores


@fieldwise_init
struct PairedTableScores(Copyable, Movable):
    """An owned X score table and the corresponding dense Y scores."""

    var x_scores: ColumnTable[DType.float64]
    var y_scores: Matrix


def _as_table_scores(
    var scores: PairedScores, names: List[String]
) raises -> PairedTableScores:
    var x_table = ColumnTable[DType.float64](scores.x_scores, names)
    return PairedTableScores(x_table^, scores^._take_y_scores())


trait CrossDecompositionTransformer(SupervisedEstimator, TransformerMixin):
    comptime PairedTransform: Movable

    def transform(
        self, X: Self.Input, y: Self.Target
    ) raises -> Self.PairedTransform:
        ...


trait PLSPredictor(RegressorMixin):
    pass
