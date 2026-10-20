"""Native dense metric kernels. See LICENSES/sklearn.txt for reference notices."""
import numojo as nm
from std.sys import llvm_intrinsic
from std.math import nan, sqrt, cbrt, abs, max, min, isnan, isfinite, sin, cos, asin
from moxie.neighbors._contracts import DenseMetric
from moxie._linalg import covariance_inverse
from moxie._dense import sum_axis2d


def metric_power(x: Float64, p: Float64) -> Float64:
    """Use accurate native algebraic primitives for common Minkowski powers."""
    if p == 0.5: return sqrt(x)
    if p == 1: return x
    if p == 1.5: return x * sqrt(x)
    if p == 2: return x * x
    if p == 3: return x * x * x
    if p == Float64(1)/3: return cbrt(x)
    if p == Float64(2)/3:
        var root = cbrt(x)
        return root * root
    return llvm_intrinsic["llvm.pow.f64", Float64](x, p)


def at[dtype: DType](x: nm.NDArray[dtype], i: Int, j: Int) raises -> Float64:
    return Float64(x.unsafe_get(i * Int(x.strides[0]) + j * Int(x.strides[1])))


def canonical_metric(name: String) raises -> String:
    if name == "l2": return "euclidean"
    if name == "l1" or name == "cityblock": return "manhattan"
    if name == "infinity": return "chebyshev"
    for accepted in ["euclidean", "sqeuclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "cosine", "correlation", "haversine", "nan_euclidean", "canberra", "braycurtis", "hamming", "jaccard", "dice", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath", "yule", "seuclidean", "mahalanobis", "precomputed"]:
        if name == accepted: return name
    raise Error("ValueError: unknown dense metric: " + name)


trait SelfQueryMetric(DenseMetric):
    """Engine-private refinement for data-derived parameters of self queries.

    Mirrors sklearn `_precompute_metric_params`, which infers parameters only
    when the query is the fitted data itself (`X is Y`).
    """

    def self_query_metric(self, fit_x: nm.NDArray[Self.dtype]) raises -> Self:
        ...

    def validate_cross_query(self) raises:
        ...

    def allows_undefined(self) -> Bool:
        ...


def inferred_seuclidean_V[dtype: DType](x: nm.NDArray[dtype]) raises -> nm.NDArray[DType.float64]:
    """Sample variances, using shared reductions and widened accumulation."""
    var xf = x.astype[DType.float64]()
    var means = sum_axis2d(xf, 0)
    for j in range(x.shape[1]):
        means.unsafe_set(j, means.unsafe_get(j) / Float64(x.shape[0]))
    var centered = xf - means
    var squares = centered * centered
    var result = sum_axis2d(squares, 0)
    for j in range(x.shape[1]):
        result.unsafe_set(j, result.unsafe_get(j) / Float64(x.shape[0] - 1))
    return result^


def inferred_mahalanobis_VI[dtype: DType](x: nm.NDArray[dtype]) raises -> nm.NDArray[DType.float64]:
    """Return sklearn `np.linalg.inv(np.cov(X.T)).T`, in Float64 like np.cov."""
    var n = x.shape[0]
    var d = x.shape[1]
    if d < 2:
        # np.cov of one feature is a 0-d array, which np.linalg.inv rejects.
        raise Error("ValueError: inferred Mahalanobis VI requires at least two features")
    var xf = x.astype[DType.float64]()
    var centered = xf - nm.mean[returned_dtype=DType.float64](xf, axis=0)
    var gram = centered.T() @ centered
    # The Gram is mathematically symmetric; mirror one triangle so rounding in
    # the product kernel cannot fail the adapter's exact symmetry contract.
    var covariance = nm.zeros[DType.float64](nm.Shape(d, d))
    for i in range(d):
        for j in range(i + 1):
            var value = at(gram, i, j) / Float64(n - 1)
            covariance.unsafe_set(i * d + j, value)
            covariance.unsafe_set(j * d + i, value)
    try:
        return covariance_inverse(covariance).T()
    except error:
        raise Error("ValueError: cannot infer Mahalanobis VI from the fitted data: " + String(error))


struct Metric[dtype_: DType](SelfQueryMetric):
    comptime dtype = Self.dtype_
    var name: String
    var p: Float64
    var missing_values: Float64
    var w: Optional[nm.NDArray[Self.dtype]]
    var V: Optional[nm.NDArray[Self.dtype]]
    var VI: Optional[nm.NDArray[Self.dtype]]
    # Data-derived precision for self queries only; never a constructor parameter.
    var _inferred_V: Optional[nm.NDArray[DType.float64]]
    var _inferred_VI: Optional[nm.NDArray[DType.float64]]

    def __init__(out self, name: String = "minkowski", p: Float64 = 2,
                 w: Optional[nm.NDArray[Self.dtype]] = None,
                 V: Optional[nm.NDArray[Self.dtype]] = None,
                 VI: Optional[nm.NDArray[Self.dtype]] = None,
                 missing_values: Float64 = nan[DType.float64]()) raises:
        self.name = canonical_metric(name)
        self.p = p
        self.missing_values = missing_values
        self.w = w.copy()
        self.V = V.copy()
        self.VI = VI.copy()
        self._inferred_VI = None
        self._inferred_V = None
        if self.name == "minkowski" and not self.w:
            if p == 1: self.name = "manhattan"
            elif p == 2: self.name = "euclidean"
            elif p == Float64("inf"): self.name = "chebyshev"
        if isnan(p) or p <= 0:
            raise Error("ValueError: p must be positive")

    def validate_features(self, n_features: Int) raises:
        comptime assert Self.dtype == DType.float32 or Self.dtype == DType.float64
        if n_features <= 0: raise Error("ValueError: empty feature space")
        if self.name == "haversine" and n_features != 2:
            raise Error("ValueError: haversine requires two features")
        if self.w:
            ref w = self.w.value()
            if self.name != "minkowski" and self.name != "wminkowski":
                raise Error("ValueError: feature weights require Minkowski")
            if w.ndim != 1 or w.shape[0] != n_features:
                raise Error("ValueError: weights must match features")
            for j in range(n_features):
                var v = w.unsafe_get(j * Int(w.strides[0]))
                if not isfinite(v) or v < 0: raise Error("ValueError: invalid feature weight")
        if self.name == "wminkowski" and not self.w:
            raise Error("ValueError: wminkowski requires weights")
        if self.name == "seuclidean" and self.V:
            ref v = self.V.value()
            if v.ndim != 1 or v.shape[0] != n_features: raise Error("ValueError: V must match features")
        if self.name == "mahalanobis":
            if not self.VI:
                # VI is inferred for self queries; covariance-form V is not a
                # brute-force parameter in sklearn either.
                if self.V: raise Error("ValueError: brute mahalanobis requires VI, not V")
                return
            ref v = self.VI.value()
            if v.ndim != 2 or v.shape[0] != n_features or v.shape[1] != n_features:
                raise Error("ValueError: VI must be a feature-square matrix")
            for i in range(n_features):
                for j in range(n_features):
                    if not isfinite(at(v, i, j)): raise Error("ValueError: nonfinite VI")

    def allows_undefined(self) -> Bool:
        return self.name == "nan_euclidean" or self.name == "correlation" or self.name == "braycurtis" or self.name == "dice" or self.name == "sokalsneath" or self.name == "seuclidean" or self.name == "mahalanobis"

    def self_query_metric(self, fit_x: nm.NDArray[Self.dtype]) raises -> Self:
        var prepared = self.copy()
        # A single fitted row has no non-self pairs, so nothing is evaluated.
        if self.name == "mahalanobis" and not self.VI and fit_x.shape[0] > 1:
            prepared._inferred_VI = inferred_mahalanobis_VI(fit_x)
        if self.name == "seuclidean" and not self.V and fit_x.shape[0] > 1:
            prepared._inferred_V = inferred_seuclidean_V(fit_x)
        return prepared^

    def validate_cross_query(self) raises:
        if self.name == "seuclidean" and not self.V:
            raise Error("ValueError: The 'V' parameter is required for the seuclidean metric when Y is passed.")
        if self.name == "mahalanobis" and not self.VI:
            raise Error("ValueError: The 'VI' parameter is required for the mahalanobis metric when Y is passed.")

    def distance(self, x: nm.NDArray[Self.dtype], x_row: Int,
                 y: nm.NDArray[Self.dtype], y_row: Int) raises -> Float64:
        var n = x.shape[1]
        if self.name == "precomputed":
            raise Error("ValueError: precomputed distances bypass metric evaluation")
        if self.name == "haversine":
            var a = sin((at(x, x_row, 0) - at(y, y_row, 0)) / 2)
            var b = sin((at(x, x_row, 1) - at(y, y_row, 1)) / 2)
            var t = a*a + cos(at(x,x_row,0))*cos(at(y,y_row,0))*b*b
            return 2 * asin(sqrt(min(Float64(1), max(Float64(0), t))))
        if self.name == "mahalanobis":
            var total = Float64(0)
            if self.VI:
                for i in range(n):
                    var left = at(x,x_row,i) - at(y,y_row,i)
                    for j in range(n):
                        total += left * at(self.VI.value(),i,j) * (at(x,x_row,j)-at(y,y_row,j))
            elif self._inferred_VI:
                for i in range(n):
                    var left = at(x,x_row,i) - at(y,y_row,i)
                    for j in range(n):
                        total += left * at(self._inferred_VI.value(),i,j) * (at(x,x_row,j)-at(y,y_row,j))
            else:
                self.validate_cross_query()
            return sqrt(total)
        var mx = Float64(0)
        var my = Float64(0)
        if self.name == "correlation":
            for j in range(n):
                mx += at(x,x_row,j) / Float64(n)
                my += at(y,y_row,j) / Float64(n)
        var total = Float64(0)
        var denominator = Float64(0)
        var xx = Float64(0)
        var yy = Float64(0)
        var tt = Float64(0)
        var tf = Float64(0)
        var ft = Float64(0)
        var ff = Float64(0)
        var common = 0
        for j in range(n):
            var a = at(x,x_row,j)
            var b = at(y,y_row,j)
            if self.name == "nan_euclidean":
                if isnan(self.missing_values):
                    if isnan(a) or isnan(b): continue
                elif a == self.missing_values or b == self.missing_values: continue
            common += 1
            var delta = abs(a-b)
            var weight = Float64(1)
            if self.w: weight = Float64(self.w.value().unsafe_get(j * Int(self.w.value().strides[0])))
            if self.name == "minkowski" or self.name == "wminkowski":
                if weight == 0: continue
                if self.p == Float64("inf"):
                    total = max(total, delta)
                elif self.name == "wminkowski":
                    total += metric_power(weight * delta, self.p)
                else:
                    total += weight * metric_power(delta, self.p)
            elif self.name == "euclidean" or self.name == "sqeuclidean" or self.name == "nan_euclidean":
                total += delta * delta
            elif self.name == "seuclidean":
                var variance: Float64
                if self.V:
                    variance = Float64(self.V.value().unsafe_get(j * Int(self.V.value().strides[0])))
                elif self._inferred_V:
                    variance = self._inferred_V.value().unsafe_get(j)
                else:
                    self.validate_cross_query()
                    raise Error("ValueError: missing variance")
                total += delta * delta / variance
            elif self.name == "manhattan": total += delta
            elif self.name == "chebyshev": total = max(total, delta)
            elif self.name == "cosine" or self.name == "correlation":
                total += (a-mx)*(b-my)
                xx += (a-mx)*(a-mx)
                yy += (b-my)*(b-my)
            elif self.name == "canberra":
                var divisor = abs(a)+abs(b)
                if divisor != 0: total += delta/divisor
            elif self.name == "braycurtis":
                total += delta
                denominator += abs(a)+abs(b)
            elif self.name == "hamming": total += Float64(1) if a != b else Float64(0)
            else:
                if a != 0 and b != 0: tt += 1
                elif a != 0: tf += 1
                elif b != 0: ft += 1
                else: ff += 1
        if self.name == "minkowski" or self.name == "wminkowski":
            return total if self.p == Float64("inf") else metric_power(total, 1/self.p)
        if self.name == "euclidean" or self.name == "seuclidean": return sqrt(total)
        if self.name == "nan_euclidean":
            return Float64("nan") if common == 0 else sqrt(total * Float64(n)/Float64(common))
        if self.name == "cosine" or self.name == "correlation":
            if xx == 0 or yy == 0:
                return Float64(1) if self.name == "cosine" else Float64("nan")
            return min(Float64(2),max(Float64(0),1-total/(sqrt(xx)*sqrt(yy))))
        if self.name == "braycurtis": return total/denominator if denominator > 0 else Float64(0)
        if self.name == "hamming": return total/Float64(n)
        var neq = tf+ft
        if self.name == "jaccard": return Float64(0) if tt+neq == 0 else neq/(tt+neq)
        if self.name == "dice": return neq/(2*tt+neq)
        if self.name == "rogerstanimoto" or self.name == "sokalmichener": return 2*neq/(Float64(n)+neq)
        if self.name == "russellrao": return (Float64(n)-tt)/Float64(n)
        if self.name == "sokalsneath": return neq/(neq+tt/2)
        if self.name == "yule": return Float64(0) if tf*ft == 0 else 2*tf*ft/(tt*ff+tf*ft)
        return total
