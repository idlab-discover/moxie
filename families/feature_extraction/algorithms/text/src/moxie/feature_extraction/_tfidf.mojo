# Adapted from scikit-learn sklearn/feature_extraction/text.py.
# Copyright (c) scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Dense term-frequency and inverse-document-frequency weighting."""

from std.math import abs, inf, isfinite, sqrt
from std.sys import llvm_intrinsic

from numojo import NDArray, Shape, zeros
from moxie._feature_metadata import FeatureNamesOut, InputFeatureCount
from moxie.utils.validation import FittedState
from moxie.feature_extraction._base import FeatureExtractor


def _log[dtype: DType](value: Scalar[dtype]) -> Scalar[dtype]:
    # Lower to the platform libm log/logf on the locked CPU toolchain.
    return llvm_intrinsic["llvm.log", Scalar[dtype], has_side_effect=False](value)


@fieldwise_init
struct TfidfTransformerParams(Copyable, Movable):
    var norm: Optional[String]
    var use_idf: Bool
    var smooth_idf: Bool
    var sublinear_tf: Bool


struct TfidfTransformer[dtype: DType = DType.float64](
    FeatureExtractor, FeatureNamesOut, InputFeatureCount, Copyable
):
    """Learn term weights from a dense document-by-term count matrix."""

    comptime output_dtype = DType.float32 if Self.dtype == DType.float32 else DType.float64
    comptime Input = NDArray[Self.dtype]
    comptime Transformed = NDArray[Self.output_dtype]
    comptime Target = Self.Transformed
    comptime Params = TfidfTransformerParams

    var norm: Optional[String]
    var use_idf: Bool
    var smooth_idf: Bool
    var sublinear_tf: Bool
    var _state: FittedState
    var _idf: Optional[Self.Transformed]

    def __init__(
        out self, *,
        norm: Optional[String] = String("l2"),
        use_idf: Bool = True,
        smooth_idf: Bool = True,
        sublinear_tf: Bool = False,
    ):
        comptime assert (
            Self.dtype == DType.float32
            or Self.dtype == DType.float64
            or Self.dtype.is_integral()
            or Self.dtype == DType.bool
        ), "TfidfTransformer requires Float32, Float64, integral or Boolean counts"
        self.norm = norm.copy()
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self._state = FittedState()
        self._idf = None

    def get_params(self, deep: Bool = True) -> Self.Params:
        return Self.Params(self.norm.copy(), self.use_idf, self.smooth_idf, self.sublinear_tf)

    def set_params[origin: MutOrigin, params_origin: ImmOrigin](
        ref[origin] self, ref[params_origin] params: Self.Params
    ) raises -> ref[self] Self:
        self.norm = params.norm.copy()
        self.use_idf = params.use_idf
        self.smooth_idf = params.smooth_idf
        self.sublinear_tf = params.sublinear_tf
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        norm: Optional[String],
        use_idf: Optional[Bool] = None,
        smooth_idf: Optional[Bool] = None,
        sublinear_tf: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        self.norm = norm.copy()
        self.use_idf = use_idf.or_else(self.use_idf)
        self.smooth_idf = smooth_idf.or_else(self.smooth_idf)
        self.sublinear_tf = sublinear_tf.or_else(self.sublinear_tf)
        return self

    def set_params[origin: MutOrigin](
        ref[origin] self, *,
        use_idf: Optional[Bool] = None,
        smooth_idf: Optional[Bool] = None,
        sublinear_tf: Optional[Bool] = None,
    ) raises -> ref[self] Self:
        var norm = self.norm.copy()
        return self.set_params(
            norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf,
        )

    def __sklearn_clone__(self) -> Self:
        return Self(
            norm=self.norm, use_idf=self.use_idf,
            smooth_idf=self.smooth_idf, sublinear_tf=self.sublinear_tf,
        )

    def __sklearn_is_fitted__(self) -> Bool:
        return self._state.is_fitted()

    def n_features_in_(self) raises -> Int:
        if not self._state.is_fitted():
            raise Error("NotFittedError: call fit before reading n_features_in_")
        return self._state.n_features_in_.value()

    def get_n_features_in(self) raises -> Int:
        return self.n_features_in_()

    def idf_(self) raises -> Self.Transformed:
        if not self._idf:
            raise Error("AttributeError: idf_ requires a successful fit with use_idf=True")
        return self._idf.value().copy()

    def _check_norm(self) raises:
        if self.norm and self.norm.value() != "l1" and self.norm.value() != "l2":
            raise Error("ValueError: norm must be 'l1', 'l2' or None")

    @staticmethod
    def _check_shape(X: Self.Input) raises:
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise Error("ValueError: expected a rank-two matrix with samples and features")

    def fit[origin: MutOrigin, x_origin: ImmOrigin](
        ref[origin] self, ref[x_origin] X: Self.Input,
        y: Optional[Self.Target] = None,
    ) raises -> ref[self] Self:
        self._check_norm()
        Self._check_shape(X)
        var rows = X.shape[0]
        var columns = X.shape[1]
        var row_stride = Int(X.strides[0])
        var column_stride = Int(X.strides[1])
        var frequency = List[Int](length=columns if self.use_idf else 0, fill=0)
        for row in range(rows):
            for column in range(columns):
                var value = X.unsafe_get(row * row_stride + column * column_stride)
                if not isfinite(value.cast[DType.float64]()):
                    raise Error("ValueError: X contains a nonfinite count")
                if self.use_idf and value != Scalar[Self.dtype](0):
                    frequency[column] += 1

        var idf: Optional[Self.Transformed] = None
        if self.use_idf:
            comptime Float = Scalar[Self.output_dtype]
            var weights = zeros[Self.output_dtype](Shape(columns))
            var smooth = Float(Int(self.smooth_idf))
            var numerator = Float(rows + Int(self.smooth_idf))
            for column in range(columns):
                var denominator = Float(frequency[column]) + smooth
                var weight = inf[Self.output_dtype]()
                if denominator != 0:
                    weight = _log(numerator / denominator) + 1
                weights.unsafe_set(column, weight)
            idf = weights^
        self._idf = idf^
        self._state.record_fit(rows, columns)
        return self

    def transform(self, X: Self.Input) raises -> Self.Transformed:
        return self.transform(X, copy=True)

    def transform(self, X: Self.Input, copy: Bool) raises -> Self.Transformed:
        """Return an owned dense array. Dense input is unchanged for either copy value."""
        _ = self.n_features_in_()
        self._check_norm()
        Self._check_shape(X)
        var columns = X.shape[1]
        self._state.check_features(columns)
        var row_stride = Int(X.strides[0])
        var column_stride = Int(X.strides[1])
        var result = zeros[Self.output_dtype](Shape(X.shape[0], columns))
        var normalize = Bool(self.norm)
        var l2 = normalize and self.norm.value() == "l2"
        var weight_terms = Bool(self._idf)
        for row in range(X.shape[0]):
            var norm_sum = Float64(0)
            var output_start = row * columns
            for column in range(columns):
                var value = X.unsafe_get(row * row_stride + column * column_stride).cast[Self.output_dtype]()
                if not isfinite(value):
                    raise Error("ValueError: X contains a nonfinite count")
                # A dense zero corresponds to an unstored term in the reference.
                if value == 0:
                    continue
                if self.sublinear_tf:
                    value = _log(value) + 1
                if weight_terms:
                    value *= self._idf.value().unsafe_get(column)
                if normalize:
                    if not isfinite(value):
                        raise Error("ValueError: weighted counts must be finite for normalization")
                    norm_sum += Float64(value * value) if l2 else abs(Float64(value))
                result.unsafe_set(output_start + column, value)
            if norm_sum != 0:
                var divisor = sqrt(norm_sum) if l2 else norm_sum
                for column in range(columns):
                    var index = output_start + column
                    var normalized = Float64(result.unsafe_get(index)) / divisor
                    result.unsafe_set(index, normalized.cast[Self.output_dtype]())
        return result^

    def fit_transform(
        mut self, X: Self.Input, y: Optional[Self.Target] = None
    ) raises -> Self.Transformed:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[String]] = None
    ) raises -> List[String]:
        var columns = self.n_features_in_()
        if input_features:
            if len(input_features.value()) != columns:
                raise Error("ValueError: input_features length differs from fitted feature count")
            return input_features.value().copy()
        var names = List[String](capacity=columns)
        for column in range(columns):
            names.append(String("x", column))
        return names^
