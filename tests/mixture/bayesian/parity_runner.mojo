"""Parity driver: run BayesianGaussianMixture on cases written by tests/parity.py.

Case file: one `key value...` record per line. Numbers are exchanged as the
decimal of their IEEE-754 Float64 bit pattern, so values round-trip exactly.
  dtype f32|f64            scalar params: n_components, covariance_type, tol,
  reg_covar, max_iter, n_init, init_params, random_state, warm_start,
  n_fits, n_draws, weight_concentration_prior_type and optional scalar priors
  (bits or "none"); arrays: `array <name> <ndim> <shape...> <bits...>` for
  X, T (test rows), mean_prior, covariance_prior.
Output: `<name> <count> <bits...>` per result, or `error <message>`.
"""
from numojo import NDArray, Shape, zeros
from std.sys import argv
from std.memory import bitcast
from moxie.mixture import BayesianGaussianMixture


def to_bits(value: Float64) -> String:
    return String(bitcast[DType.int64](value))


def from_bits(text: String) raises -> Float64:
    return bitcast[DType.float64](Int64(Int(text)))


struct Case(Movable):
    var keys: List[String]
    var values: List[String]
    var array_names: List[String]
    var array_shapes: List[List[Int]]
    var array_values: List[List[Float64]]

    def __init__(out self, text: String) raises:
        self.keys = List[String]()
        self.values = List[String]()
        self.array_names = List[String]()
        self.array_shapes = List[List[Int]]()
        self.array_values = List[List[Float64]]()
        for raw_line in text.split("\n"):
            var line = String(raw_line)
            if line.byte_length() == 0:
                continue
            var parts = line.split(" ")
            if String(parts[0]) == "array":
                self.array_names.append(String(parts[1]))
                var ndim = Int(String(parts[2]))
                var shape = List[Int]()
                var count = 1
                for axis in range(ndim):
                    shape.append(Int(String(parts[3 + axis])))
                    count *= shape[axis]
                var data = List[Float64]()
                for i in range(count):
                    data.append(from_bits(String(parts[3 + ndim + i])))
                self.array_shapes.append(shape^)
                self.array_values.append(data^)
            else:
                self.keys.append(String(parts[0]))
                self.values.append(String(parts[1]))

    def get(self, key: String, default: String) -> String:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.values[i]
        return default

    def has_array(self, name: String) -> Bool:
        for i in range(len(self.array_names)):
            if self.array_names[i] == name:
                return True
        return False

    def array[dtype: DType](self, name: String) raises -> NDArray[dtype]:
        for i in range(len(self.array_names)):
            if self.array_names[i] == name:
                var shape = self.array_shapes[i].copy()
                var out: NDArray[dtype]
                if len(shape) == 1:
                    out = zeros[dtype](Shape(shape[0]))
                elif len(shape) == 2:
                    out = zeros[dtype](Shape(shape[0], shape[1]))
                else:
                    out = zeros[dtype](Shape(shape[0], shape[1], shape[2]))
                for j in range(len(self.array_values[i])):
                    out.unsafe_set(j, self.array_values[i][j].cast[dtype]())
                return out^
        raise Error("missing array " + name)


def emit[dtype: DType](mut buffer: String, name: String, x: NDArray[dtype]) raises:
    buffer += name + " " + String(x.size)
    for i in range(x.size):
        buffer += " " + to_bits(Float64(x.unsafe_get(i)))
    buffer += "\n"


def emit_scalar(mut buffer: String, name: String, value: Float64):
    buffer += name + " 1 " + to_bits(value) + "\n"


def optional_scalar(spec: Case, key: String) raises -> Optional[Float64]:
    var raw = spec.get(key, "none")
    if raw == "none":
        return None
    return from_bits(raw)


def run[dtype: DType](spec: Case) raises -> String:
    var X = spec.array[dtype]("X")
    var T = spec.array[dtype]("T")
    var mean_prior = Optional[NDArray[dtype]](None)
    var covariance_prior = Optional[NDArray[dtype]](None)
    if spec.has_array("mean_prior"):
        mean_prior = spec.array[dtype]("mean_prior")
    if spec.has_array("covariance_prior"):
        covariance_prior = spec.array[dtype]("covariance_prior")
    var model = BayesianGaussianMixture[dtype](
        n_components=Int(spec.get("n_components", "1")),
        covariance_type=spec.get("covariance_type", "full"),
        tol=from_bits(spec.get("tol", to_bits(1e-3))),
        reg_covar=from_bits(spec.get("reg_covar", to_bits(1e-6))),
        max_iter=Int(spec.get("max_iter", "100")),
        n_init=Int(spec.get("n_init", "1")),
        init_params=spec.get("init_params", "kmeans"),
        weight_concentration_prior_type=spec.get("weight_concentration_prior_type", "dirichlet_process"),
        weight_concentration_prior=optional_scalar(spec, "weight_concentration_prior"),
        mean_precision_prior=optional_scalar(spec, "mean_precision_prior"),
        mean_prior=mean_prior^,
        degrees_of_freedom_prior=optional_scalar(spec, "degrees_of_freedom_prior"),
        covariance_prior=covariance_prior^,
        random_state=Int(spec.get("random_state", "0")),
        warm_start=spec.get("warm_start", "0") == "1",
    )
    var out = String("")
    var n_fits = Int(spec.get("n_fits", "1"))
    var labels = zeros[DType.int64](Shape(0))
    for fit in range(n_fits):
        labels = model.fit_predict(X)
        emit_scalar(out, "fit_lower_bound_" + String(fit), model.lower_bound_)
    var label_values = zeros[DType.float64](Shape(labels.size))
    for i in range(labels.size):
        label_values.unsafe_set(i, Float64(labels.unsafe_get(i)))
    emit(out, "fit_predict", label_values)
    emit(out, "weights_", model.weights_)
    emit(out, "means_", model.means_)
    emit(out, "covariances_", model.covariances_)
    emit(out, "precisions_", model.precisions_)
    emit(out, "precisions_cholesky_", model.precisions_cholesky_)
    for i in range(len(model.weight_concentration_)):
        emit(out, "weight_concentration_" + String(i), model.weight_concentration_[i])
    emit(out, "mean_precision_", model.mean_precision_)
    emit(out, "degrees_of_freedom_", model.degrees_of_freedom_)
    emit(out, "mean_prior_", model.mean_prior_)
    emit(out, "covariance_prior_", model.covariance_prior_)
    emit_scalar(out, "weight_concentration_prior_", model.weight_concentration_prior_)
    emit_scalar(out, "mean_precision_prior_", model.mean_precision_prior_)
    emit_scalar(out, "degrees_of_freedom_prior_", model.degrees_of_freedom_prior_)
    emit_scalar(out, "converged_", 1.0 if model.converged_ else 0.0)
    emit_scalar(out, "n_iter_", Float64(model.n_iter_))
    emit_scalar(out, "lower_bound_", model.lower_bound_)
    var bounds = zeros[DType.float64](Shape(len(model.lower_bounds_)))
    for i in range(len(model.lower_bounds_)):
        bounds.unsafe_set(i, model.lower_bounds_[i])
    emit(out, "lower_bounds_", bounds)
    var predicted = model.predict(T)
    var predicted_values = zeros[DType.float64](Shape(predicted.size))
    for i in range(predicted.size):
        predicted_values.unsafe_set(i, Float64(predicted.unsafe_get(i)))
    emit(out, "predict", predicted_values)
    emit(out, "predict_proba", model.predict_proba(T))
    emit(out, "score_samples", model.score_samples(T))
    emit_scalar(out, "score", model.score(T))
    var draws = Int(spec.get("n_draws", "0"))
    if draws > 0:
        var drawn = model.sample(draws)
        emit(out, "sample_X", drawn[0])
        var sample_labels = zeros[DType.float64](Shape(drawn[1].size))
        for i in range(drawn[1].size):
            sample_labels.unsafe_set(i, Float64(drawn[1].unsafe_get(i)))
        emit(out, "sample_y", sample_labels)
    return out^


def main() raises:
    var args = argv()
    var text: String
    with open(args[1], "r") as f:
        text = f.read()
    var spec = Case(text)
    var result: String
    try:
        if spec.get("dtype", "f64") == "f32":
            result = run[DType.float32](spec)
        else:
            result = run[DType.float64](spec)
    except error:
        result = "error " + String(error) + "\n"
    with open(args[2], "w") as f:
        f.write(result)
