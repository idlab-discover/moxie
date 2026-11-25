"""Typed model-selection integration for isotonic regression."""

from std.utils import Variant

from moxie.isotonic import Increasing, IsotonicRegression
from moxie.model_selection import (
    IndexArray,
    IndexSplit,
    KFold,
    SearchAdapter,
    SearchScorer,
    select_rows,
)
from numojo import NDArray


comptime IsotonicSearchValue = Variant[Float64, Bool, String, NoneType]


struct IsotonicScorer[dtype: DType = DType.float64, list_inputs: Bool = False](
    SearchScorer
):
    """Use the estimator's R² score, with optional per-sample weights."""

    comptime Estimator = IsotonicRegression[Self.dtype]
    comptime Vector = List[Scalar[Self.dtype]]
    comptime Dense = NDArray[Self.dtype]
    comptime Input = Self.Vector if Self.list_inputs else Self.Dense
    comptime Target = Self.Input
    comptime Metadata = Optional[Self.Target]

    def __init__(out self):
        pass

    def score(
        self,
        estimator: Self.Estimator,
        X: Self.Input,
        y: Self.Target,
        metadata: Self.Metadata,
    ) raises -> Float64:
        # The compiler does not narrow conditional associated types inside comptime if.
        comptime if Self.list_inputs:
            return estimator.score(
                rebind[Self.Vector](X), rebind[Self.Vector](y),
                sample_weight=rebind[Optional[Self.Vector]](metadata),
            )
        else:
            return estimator.score(
                rebind[Self.Dense](X), rebind[Self.Dense](y),
                sample_weight=rebind[Optional[Self.Dense]](metadata),
            )


struct IsotonicCVAdapter[dtype: DType = DType.float64, list_inputs: Bool = False](
    SearchAdapter
):
    """Slice isotonic data for cross-validation and typed parameter search."""

    comptime Scorer = IsotonicScorer[Self.dtype, Self.list_inputs]
    comptime Value = IsotonicSearchValue
    comptime _Vector = Self.Scorer.Vector
    comptime _Dense = Self.Scorer.Dense

    def __init__(out self):
        pass

    def n_samples(self, X: Self.Scorer.Input, y: Self.Scorer.Target) raises -> Int:
        var n: Int
        var targets: Int
        comptime if Self.list_inputs:
            n = len(rebind[Self._Vector](X))
            targets = len(rebind[Self._Vector](y))
        else:
            ref features = rebind[Self._Dense](X)
            ref target = rebind[Self._Dense](y)
            if features.ndim != 1 and (features.ndim != 2 or features.shape[1] != 1):
                raise Error("ValueError: X should be a 1d array or 2d array with 1 feature")
            if target.ndim != 1:
                raise Error("ValueError: y should be a 1d array")
            n = features.shape[0]
            targets = target.shape[0]
        if n == 0 or n != targets:
            raise Error("ValueError: X and y must have the same positive sample count")
        return n

    def input_rows(
        self, X: Self.Scorer.Input, rows: IndexArray, train: IndexArray
    ) raises -> Self.Scorer.Input:
        comptime if Self.list_inputs:
            return rebind_var[Self.Scorer.Input](select_rows(rebind[Self._Vector](X), rows))
        else:
            return rebind_var[Self.Scorer.Input](select_rows(rebind[Self._Dense](X), rows))

    def target_rows(
        self, y: Self.Scorer.Target, rows: IndexArray
    ) raises -> Self.Scorer.Target:
        return self.input_rows(y, rows, rows)

    def metadata_rows(
        self, metadata: Self.Scorer.Metadata, rows: IndexArray
    ) raises -> Self.Scorer.Metadata:
        if not metadata:
            return None
        ref weights = metadata.value()
        comptime if not Self.list_inputs:
            if rebind[Self._Dense](weights).ndim != 1:
                raise Error("ValueError: sample_weight should be a 1d array")
        return self.target_rows(weights, rows)

    def fit(
        self,
        mut estimator: Self.Scorer.Estimator,
        X: Self.Scorer.Input,
        y: Self.Scorer.Target,
        metadata: Self.Scorer.Metadata,
    ) raises:
        comptime if Self.list_inputs:
            ref features = rebind[Self._Vector](X)
            ref target = rebind[Self._Vector](y)
            if metadata:
                _ = estimator.fit(
                    features, target, sample_weight=rebind[Self._Vector](metadata.value()),
                )
            else:
                _ = estimator.fit(features, target)
        else:
            ref features = rebind[Self._Dense](X)
            ref target = rebind[Self._Dense](y)
            if metadata:
                _ = estimator.fit(
                    features, target, sample_weight=rebind[Self._Dense](metadata.value()),
                )
            else:
                _ = estimator.fit(features, target)

    def configure(
        self, mut estimator: Self.Scorer.Estimator, params: Dict[String, Self.Value]
    ) raises:
        var config = estimator.get_params()
        for entry in params.items():
            ref name = entry.key
            ref value = entry.value
            if name == "y_min" or name == "y_max":
                var bound: Optional[Float64] = None
                if value.isa[Float64]():
                    bound = value[Float64]
                elif not value.isa[NoneType]():
                    raise Error("ValueError: " + name + " requires Float64 or None")
                if name == "y_min":
                    config.y_min = bound
                else:
                    config.y_max = bound
            elif name == "increasing":
                if value.isa[Bool]():
                    config.increasing = Increasing(value[Bool])
                elif value.isa[String]():
                    config.increasing = Increasing(value[String].copy())
                else:
                    raise Error("ValueError: increasing requires Bool or String")
            elif name == "out_of_bounds":
                if not value.isa[String]():
                    raise Error("ValueError: out_of_bounds requires String")
                config.out_of_bounds = value[String].copy()
            else:
                raise Error("ValueError: invalid isotonic parameter " + name)
        _ = estimator.set_params(config)

    def default_scorer(self) -> Self.Scorer:
        return Self.Scorer()

    def default_cv(
        self, X: Self.Scorer.Input, y: Self.Scorer.Target
    ) raises -> List[IndexSplit]:
        var cv = KFold()
        var cursor = cv.split(self.n_samples(X, y))
        var folds = List[IndexSplit](capacity=cv.n_splits)
        for _ in range(cv.n_splits):
            var split = cursor.next()
            folds.append(split.take())
        return folds^
