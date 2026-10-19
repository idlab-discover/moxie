# Adapted from scikit-learn; SPDX-License-Identifier: BSD-3-Clause
"""Native curve displays with owned data, inspectable artists and SVG output."""
from numojo import NDArray, Shape, Item
from moxie._dense import sum_axis2d
from moxie.model_selection._indices import IndexArray, IndexSplit
from moxie.model_selection._evaluation import ParameterAdapter
from moxie.model_selection._validation import CVAdapter, learning_curve, validation_curve
from moxie.model_selection._results import ErrorPolicy
from std.math import sqrt, isfinite, log10, abs
from std.memory import bitcast

comptime Dense = NDArray[DType.float64]


@fieldwise_init
struct CurveArtist(Copyable, Movable):
    var label: String
    var x: Dense
    var mean: Dense
    var std: Dense
    var color: String
    var alpha: Float64


@fieldwise_init
struct CurveAxes(Copyable, Movable):
    var xlabel: String
    var ylabel: String
    var xscale: String


@fieldwise_init
struct CurveFigure(Copyable, Movable):
    var svg: String

    def savefig(self, path: String) raises:
        with open(path, "w") as output:
            output.write(self.svg)


def _xml(value: String) -> String:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")


def _moments(scores: Dense, negate: Bool) raises -> Tuple[Dense, Dense]:
    var mean = sum_axis2d(scores, 1)
    var residuals = Dense(scores.shape.copy())
    for i in range(scores.shape[0]):
        mean[Item(i)] = mean[Item(i)] / Float64(scores.shape[1])
        for j in range(scores.shape[1]):
            var delta = scores[Item(i, j)] - mean[Item(i)]
            residuals[Item(i, j)] = delta * delta
    var deviation = sum_axis2d(residuals, 1)
    for i in range(scores.shape[0]):
        deviation[Item(i)] = sqrt(deviation[Item(i)] / Float64(scores.shape[1]))
        if negate:
            mean[Item(i)] = -mean[Item(i)]
    return (mean^, deviation^)


def _scale(x: Dense) raises -> String:
    var ordered = x.copy()
    for i in range(1, x.size):
        var j = i
        while j > 0 and ordered[Item(j)] < ordered[Item(j - 1)]:
            var value = ordered[Item(j)]
            ordered[Item(j)] = ordered[Item(j - 1)]
            ordered[Item(j - 1)] = value
            j -= 1
    var smallest = ordered[Item(1)] - ordered[Item(0)]
    var largest = smallest
    for i in range(2, x.size):
        var gap = ordered[Item(i)] - ordered[Item(i - 1)]
        smallest = min(smallest, gap)
        largest = max(largest, gap)
    if largest > 5 * smallest:
        return "symlog" if ordered[Item(0)] <= 0 else "log"
    return "linear"


def _scaled(x: Float64, scale: String) -> Float64:
    if scale == "log":
        return log10(x)
    if scale == "symlog":
        # Matplotlib defaults: base 10, linthresh 2, linscale 1.
        var magnitude = abs(x)
        var value = magnitude / 0.9 if magnitude <= 2 else 2 * (1 / 0.9 + log10(magnitude / 2))
        return -value if x < 0 else value
    return x


def _svg(artists: List[CurveArtist], axes: CurveAxes, style: Optional[String]) raises -> String:
    var xlo = Float64(1e300)
    var xhi = Float64(-1e300)
    var ylo = Float64(1e300)
    var yhi = Float64(-1e300)
    for artist in artists:
        for i in range(artist.x.size):
            var x = _scaled(artist.x[Item(i)], axes.xscale)
            xlo = min(xlo, x)
            xhi = max(xhi, x)
            var y = artist.mean[Item(i)]
            var spread = artist.std[Item(i)] if style else Float64(0)
            if isfinite(y) and isfinite(spread):
                ylo = min(ylo, y - spread)
                yhi = max(yhi, y + spread)
    if ylo > yhi:
        ylo = 0
        yhi = 1
    if ylo == yhi:
        ylo -= 0.5
        yhi += 0.5
    var padding = (yhi - ylo) * 0.05
    ylo -= padding
    yhi += padding
    if xlo == xhi:
        xlo -= 0.5
        xhi += 0.5
    var svg = String('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="500" viewBox="0 0 800 500"><rect width="800" height="500" fill="white"/><g font-family="sans-serif" font-size="14"><path d="M80 35V420H760" fill="none" stroke="#333"/>')
    svg += '<text x="420" y="480" text-anchor="middle">' + _xml(axes.xlabel) + '</text><text transform="translate(20 230) rotate(-90)" text-anchor="middle">' + _xml(axes.ylabel) + '</text>'
    for tick in range(6):
        var y = ylo + (yhi - ylo) * Float64(tick) / 5
        var pixel = 420 - Float64(tick) * 77
        svg += '<text x="72" y="' + String(pixel + 5) + '" text-anchor="end">' + String(y) + '</text>'
    for a in range(len(artists)):
        var artist = artists[a].copy()
        var color = _xml(artist.color)
        svg += '<text x="' + String(520 + a * 110) + '" y="25" fill="' + color + '">' + _xml(artist.label) + '</text>'
        var path = String()
        var connected = False
        for i in range(artist.x.size):
            var px = 80 + 680 * (_scaled(artist.x[Item(i)], axes.xscale) - xlo) / (xhi - xlo)
            var y = artist.mean[Item(i)]
            if a == 0:
                svg += '<text x="' + String(px) + '" y="445" text-anchor="middle">' + String(artist.x[Item(i)]) + '</text>'
            if not isfinite(y):
                connected = False
                continue
            var py = 420 - 385 * (y - ylo) / (yhi - ylo)
            path += ("L" if connected else "M") + String(px) + " " + String(py)
            connected = True
            var spread = artist.std[Item(i)]
            if style and style.value() == "errorbar" and isfinite(spread):
                var lo = 420 - 385 * (y - spread - ylo) / (yhi - ylo)
                var hi = 420 - 385 * (y + spread - ylo) / (yhi - ylo)
                svg += '<path d="M' + String(px) + ' ' + String(lo) + 'V' + String(hi) + '" stroke="' + color + '"/>'
            if style and style.value() == "fill_between" and i > 0:
                var previous = artist.mean[Item(i - 1)]
                var previous_std = artist.std[Item(i - 1)]
                if isfinite(spread) and isfinite(previous) and isfinite(previous_std):
                    var prevx = 80 + 680 * (_scaled(artist.x[Item(i - 1)], axes.xscale) - xlo) / (xhi - xlo)
                    var polygon = String(prevx) + ',' + String(420 - 385 * (previous - previous_std - ylo) / (yhi - ylo)) + ' ' + String(px) + ',' + String(420 - 385 * (y - spread - ylo) / (yhi - ylo)) + ' ' + String(px) + ',' + String(420 - 385 * (y + spread - ylo) / (yhi - ylo)) + ' ' + String(prevx) + ',' + String(420 - 385 * (previous + previous_std - ylo) / (yhi - ylo))
                    svg += '<polygon points="' + polygon + '" fill="' + color + '" opacity="' + String(artist.alpha) + '"/>'
        svg += '<path d="' + path + '" fill="none" stroke="' + color + '" stroke-width="2"/>'
    return svg + '</g></svg>'


def _plot(x: Dense, train: Dense, test: Dense, xlabel: String, ylabel: String, negate: Bool, score_type: String, style: Optional[String], train_color: String, test_color: String, alpha: Float64) raises -> Tuple[List[CurveArtist], CurveAxes, CurveFigure]:
    if score_type != "train" and score_type != "test" and score_type != "both":
        raise Error("ValueError: Unknown score_type")
    if style and style.value() != "fill_between" and style.value() != "errorbar":
        raise Error("ValueError: Unknown std_display_style")
    if not isfinite(alpha) or alpha < 0 or alpha > 1:
        raise Error("ValueError: alpha must be in [0, 1]")
    if x.ndim != 1 or x.size < 2 or train.ndim != 2 or test.ndim != 2:
        raise Error("ValueError: display needs at least two x values and rank-two scores")
    if train.shape[0] != x.size or test.shape[0] != x.size or train.shape[1] != test.shape[1] or train.shape[1] == 0:
        raise Error("ValueError: score matrices must have shape (x values, nonempty folds)")
    for i in range(x.size):
        if not isfinite(x[Item(i)]):
            raise Error("ValueError: x values must be finite")
    var artists = List[CurveArtist]()
    if score_type != "test":
        var values = _moments(train, negate)
        artists.append(CurveArtist("Train", x.copy(), values[0].copy(), values[1].copy(), train_color.copy(), alpha))
    if score_type != "train":
        var values = _moments(test, negate)
        artists.append(CurveArtist("Test", x.copy(), values[0].copy(), values[1].copy(), test_color.copy(), alpha))
    var axes = CurveAxes(xlabel.copy(), ylabel.copy(), _scale(x))
    var figure = CurveFigure(_svg(artists, axes, style))
    return (artists^, axes^, figure^)


struct LearningCurveDisplay(Movable):
    var train_sizes: IndexArray
    var train_scores: Dense
    var test_scores: Dense
    var score_name: Optional[String]
    var lines_: Optional[List[CurveArtist]]
    var errorbar_: Optional[List[CurveArtist]]
    var fill_between_: Optional[List[CurveArtist]]
    var ax_: Optional[CurveAxes]
    var figure_: Optional[CurveFigure]

    def __init__(out self, *, train_sizes: IndexArray, train_scores: Dense, test_scores: Dense, score_name: Optional[String] = None):
        self.train_sizes = train_sizes.copy()
        self.train_scores = train_scores.copy()
        self.test_scores = test_scores.copy()
        self.score_name = score_name.copy()
        self.lines_ = None
        self.errorbar_ = None
        self.fill_between_ = None
        self.ax_ = None
        self.figure_ = None

    def plot(mut self, *, negate_score: Bool = False, score_name: Optional[String] = None, score_type: String = "both", std_display_style: Optional[String] = String("fill_between"), train_color: String = "#1f77b4", test_color: String = "#ff7f0e", alpha: Float64 = 0.5) raises:
        var x = Dense(Shape(self.train_sizes.size))
        for i in range(x.size):
            x[Item(i)] = Float64(self.train_sizes[Item(i)])
        var ylabel = String("None")
        if score_name:
            ylabel = score_name.value().copy()
        elif self.score_name:
            ylabel = self.score_name.value().copy()
        var result = _plot(x, self.train_scores, self.test_scores, "Number of samples in the training set", ylabel, negate_score, score_type, std_display_style, train_color, test_color, alpha)
        self.lines_ = None
        self.errorbar_ = None
        self.fill_between_ = None
        if std_display_style and std_display_style.value() == "errorbar":
            self.errorbar_ = result[0].copy()
        else:
            self.lines_ = result[0].copy()
            if std_display_style:
                self.fill_between_ = result[0].copy()
        self.ax_ = result[1].copy()
        self.figure_ = result[2].copy()

    @staticmethod
    def from_estimator[A: CVAdapter, size_dtype: DType = DType.float64](estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target, adapter: A, var scoring: A.Scorer, cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata, *, train_sizes: Optional[NDArray[size_dtype]] = None, shuffle: Bool = False, random_state: Optional[Int] = None, exploit_incremental_learning: Bool = False, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))), score_name: Optional[String] = None, negate_score: Bool = False, score_type: String = "both", std_display_style: Optional[String] = String("fill_between")) raises -> Self where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
        var result = learning_curve[A, size_dtype](estimator, X, y, adapter, scoring^, cv, fit_metadata, score_metadata, train_sizes=train_sizes.copy(), shuffle=shuffle, random_state=random_state, exploit_incremental_learning=exploit_incremental_learning, error_score=error_score)
        var name = score_name.copy()
        if not name:
            name = String("Negative score" if negate_score else "Score")
        var display = Self(train_sizes=result.train_sizes, train_scores=result.train_scores, test_scores=result.test_scores, score_name=name^)
        display.plot(negate_score=negate_score, score_type=score_type, std_display_style=std_display_style)
        return display^


struct ValidationCurveDisplay(Movable):
    var param_range: Dense
    var param_name: String
    var train_scores: Dense
    var test_scores: Dense
    var score_name: Optional[String]
    var lines_: Optional[List[CurveArtist]]
    var errorbar_: Optional[List[CurveArtist]]
    var fill_between_: Optional[List[CurveArtist]]
    var ax_: Optional[CurveAxes]
    var figure_: Optional[CurveFigure]

    def __init__(out self, *, param_name: String, param_range: Dense, train_scores: Dense, test_scores: Dense, score_name: Optional[String] = None):
        self.param_range = param_range.copy()
        self.param_name = param_name.copy()
        self.train_scores = train_scores.copy()
        self.test_scores = test_scores.copy()
        self.score_name = score_name.copy()
        self.lines_ = None
        self.errorbar_ = None
        self.fill_between_ = None
        self.ax_ = None
        self.figure_ = None

    def plot(mut self, *, negate_score: Bool = False, score_name: Optional[String] = None, score_type: String = "both", std_display_style: Optional[String] = String("fill_between"), train_color: String = "#1f77b4", test_color: String = "#ff7f0e", alpha: Float64 = 0.5) raises:
        var x = Dense(Shape(self.param_range.size))
        for i in range(x.size):
            x[Item(i)] = Float64(self.param_range[Item(i)])
        var ylabel = String("None")
        if score_name:
            ylabel = score_name.value().copy()
        elif self.score_name:
            ylabel = self.score_name.value().copy()
        var result = _plot(x, self.train_scores, self.test_scores, self.param_name, ylabel, negate_score, score_type, std_display_style, train_color, test_color, alpha)
        self.lines_ = None
        self.errorbar_ = None
        self.fill_between_ = None
        if std_display_style and std_display_style.value() == "errorbar":
            self.errorbar_ = result[0].copy()
        else:
            self.lines_ = result[0].copy()
            if std_display_style:
                self.fill_between_ = result[0].copy()
        self.ax_ = result[1].copy()
        self.figure_ = result[2].copy()

    @staticmethod
    def from_estimator[A: CVAdapter & ParameterAdapter](estimator: A.Scorer.Estimator, X: A.Scorer.Input, y: A.Scorer.Target, adapter: A, var scoring: A.Scorer, cv: List[IndexSplit], fit_metadata: A.Scorer.Metadata, score_metadata: A.Scorer.Metadata, *, param_name: String, param_range: List[A.Value], x_values: Dense, error_score: ErrorPolicy = ErrorPolicy(False, bitcast[DType.float64](UInt64(0x7ff8000000000000))), score_name: Optional[String] = None, negate_score: Bool = False, score_type: String = "both", std_display_style: Optional[String] = String("fill_between")) raises -> Self where conforms_to(A.Scorer.Input, Deinitable) and conforms_to(A.Scorer.Target, Deinitable) and conforms_to(A.Scorer.Metadata, Deinitable) and conforms_to(A.Scorer, Deinitable):
        if x_values.ndim != 1 or x_values.size != len(param_range):
            raise Error("ValueError: x_values must match the typed param_range")
        var result = validation_curve[A](estimator, X, y, adapter, scoring^, cv, fit_metadata, score_metadata, param_name=param_name, param_range=param_range, error_score=error_score)
        var name = score_name.copy()
        if not name:
            name = String("Negative score" if negate_score else "Score")
        var display = Self(param_name=param_name, param_range=x_values, train_scores=result[0], test_scores=result[1], score_name=name^)
        display.plot(negate_score=negate_score, score_type=score_type, std_display_style=std_display_style)
        return display^
