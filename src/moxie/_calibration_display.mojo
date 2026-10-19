# SPDX-License-Identifier: BSD-3-Clause
"""Native calibration display data and SVG reliability diagrams."""
import numojo as nm
from std.math import isfinite
from std.utils import Variant
from ._calibration_algorithms import Array, vector, matrix
from ._calibration_curve import calibration_curve


trait CalibrationDisplayEstimator(Movable):
    """Typed access to probabilities, independently of preferred decision scores.
    """

    def calibration_probabilities(self, X: Array) raises -> Array:
        ...

    def class_labels(self) raises -> Array:
        ...

    def display_name(self) -> String:
        return "Classifier"


@fieldwise_init
struct CalibrationLine(Copyable, Movable):
    var x: Array
    var y: Array
    var label: Optional[String]
    var color: String
    var alpha: Float64
    var marker: String
    var linestyle: String


def _xml(value: String) -> String:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


struct CalibrationAxes(Copyable, Movable):
    """An owned reliability-diagram axes; pass it mutably to overlay curves."""

    var lines: List[CalibrationLine]
    var reference_line: Bool
    var xlabel: String
    var ylabel: String

    def __init__(out self):
        self.lines = List[CalibrationLine]()
        self.reference_line = False
        self.xlabel = "Mean predicted probability"
        self.ylabel = "Fraction of positives"

    def legend_labels(self) -> List[String]:
        var result = List[String]()
        if self.reference_line:
            result.append("Perfectly calibrated")
        for line in self.lines:
            if line.label:
                result.append(line.label.value().copy())
        return result^

    def to_svg(self) raises -> String:
        var svg = String(
            '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="620"'
            ' viewBox="0 0 720 620" role="img"><title>Calibration'
            ' curve</title><rect width="720" height="620" fill="white"/><g'
            ' font-family="sans-serif" font-size="14">'
        )
        for i in range(6):
            var value = Float64(i) / 5
            var x = 100 + value * 540
            var y = 520 - value * 440
            svg += (
                '<path d="M'
                + String(x)
                + " 80 V520 M100 "
                + String(y)
                + ' H640" stroke="#e5e7eb" fill="none"/>'
            )
            svg += (
                '<text x="'
                + String(x)
                + '" y="545" text-anchor="middle">'
                + String(value)
                + "</text>"
            )
            svg += (
                '<text x="86" y="'
                + String(y + 5)
                + '" text-anchor="end">'
                + String(value)
                + "</text>"
            )
        svg += '<path d="M100 80 V520 H640" stroke="#111827" fill="none"/>'
        svg += (
            '<text x="370" y="585" text-anchor="middle">'
            + _xml(self.xlabel)
            + "</text>"
        )
        svg += (
            '<text transform="translate(25 300) rotate(-90)"'
            ' text-anchor="middle">'
            + _xml(self.ylabel)
            + "</text>"
        )
        if self.reference_line:
            svg += (
                '<path d="M100 520 L640 80" stroke="black" stroke-dasharray="2'
                ' 5" fill="none" data-reference="true"/>'
            )
        for line in self.lines:
            var points = String("")
            for i in range(line.x.size):
                var x = 100 + 540 * line.x.item(i)
                var y = 520 - 440 * line.y.item(i)
                points += String(x) + "," + String(y) + " "
                if line.marker == "s":
                    svg += (
                        '<rect x="'
                        + String(x - 3)
                        + '" y="'
                        + String(y - 3)
                        + '" width="6" height="6" fill="'
                        + _xml(line.color)
                        + '" opacity="'
                        + String(line.alpha)
                        + '"/>'
                    )
                elif line.marker == "o":
                    svg += (
                        '<circle cx="'
                        + String(x)
                        + '" cy="'
                        + String(y)
                        + '" r="3" fill="'
                        + _xml(line.color)
                        + '" opacity="'
                        + String(line.alpha)
                        + '"/>'
                    )
            if line.linestyle != "":
                var dash = String("")
                if line.linestyle == "--":
                    dash = ' stroke-dasharray="7 5"'
                elif line.linestyle == ":":
                    dash = ' stroke-dasharray="2 5"'
                svg += (
                    '<polyline points="'
                    + points
                    + '" stroke="'
                    + _xml(line.color)
                    + '" stroke-width="2" fill="none" opacity="'
                    + String(line.alpha)
                    + '"'
                    + dash
                    + "/>"
                )
        var legend = self.legend_labels()
        if len(legend) > 0:
            var top = 510 - 24 * len(legend)
            svg += (
                '<rect x="390" y="'
                + String(top - 17)
                + '" width="245" height="'
                + String(24 * len(legend) + 8)
                + '" fill="white" stroke="#d1d5db"/>'
            )
            var row = 0
            if self.reference_line:
                svg += (
                    '<path d="M400 '
                    + String(top - 5)
                    + ' H425" stroke="black" stroke-dasharray="2 5"/>'
                )
                svg += (
                    '<text x="435" y="'
                    + String(top)
                    + '">Perfectly calibrated</text>'
                )
                row += 1
            for line in self.lines:
                if line.label:
                    var y = top + row * 24
                    svg += (
                        '<path d="M400 '
                        + String(y - 5)
                        + ' H425" stroke="'
                        + _xml(line.color)
                        + '" stroke-width="2"/>'
                    )
                    svg += (
                        '<text x="435" y="'
                        + String(y)
                        + '">'
                        + _xml(line.label.value())
                        + "</text>"
                    )
                    row += 1
        return svg + "</g></svg>"


@fieldwise_init
struct CalibrationFigure(Copyable, Movable):
    """An owned snapshot of diagram axes, exportable without Python."""

    var axes: CalibrationAxes

    def savefig(self, path: String) raises:
        if not path.endswith(".svg"):
            raise Error(
                "ValueError: native calibration figures support .svg output"
            )
        var svg = self.axes.to_svg()
        with open(path, "w") as file:
            file.write(svg)


struct CalibrationDisplay(Copyable, Movable):
    var prob_true: Array
    var prob_pred: Array
    var y_prob: Array
    var estimator_name: Optional[String]
    var pos_label: Optional[Float64]
    var _pos_label_text: String
    var line_: Optional[CalibrationLine]
    var ax_: Optional[CalibrationAxes]
    var figure_: Optional[CalibrationFigure]

    def __init__(
        out self,
        prob_true: Array,
        prob_pred: Array,
        y_prob: Array,
        *,
        estimator_name: Optional[String] = None,
        pos_label: Optional[Float64] = None,
    ):
        self.prob_true = prob_true.copy()
        self.prob_pred = prob_pred.copy()
        self.y_prob = y_prob.copy()
        self.estimator_name = estimator_name.copy()
        self.pos_label = pos_label
        self._pos_label_text = String(pos_label.value()) if pos_label else ""
        self.line_ = None
        self.ax_ = None
        self.figure_ = None

    def plot(
        mut self,
        *,
        name: Optional[String] = None,
        ref_line: Bool = True,
        color: String = "#2563eb",
        alpha: Float64 = 1,
        marker: String = "s",
        linestyle: String = "-",
    ) raises:
        var axes = CalibrationAxes()
        self.plot(
            ax=axes,
            name=name,
            ref_line=ref_line,
            color=color,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )

    def plot(
        mut self,
        *,
        mut ax: CalibrationAxes,
        name: Optional[String] = None,
        ref_line: Bool = True,
        color: String = "#2563eb",
        alpha: Float64 = 1,
        marker: String = "s",
        linestyle: String = "-",
    ) raises:
        vector(self.prob_true)
        vector(self.prob_pred)
        if self.prob_true.size != self.prob_pred.size:
            raise Error("ValueError: curve coordinates must have equal lengths")
        if not isfinite(alpha) or alpha < 0 or alpha > 1:
            raise Error("ValueError: alpha must be in [0, 1]")
        if marker != "s" and marker != "o" and marker != "":
            raise Error("ValueError: supported markers are s, o or empty")
        if (
            linestyle != "-"
            and linestyle != "--"
            and linestyle != ":"
            and linestyle != ""
        ):
            raise Error(
                "ValueError: supported line styles are -, --, : or empty"
            )
        var label = name.copy() if name else self.estimator_name.copy()
        var line = CalibrationLine(
            self.prob_pred.copy(),
            self.prob_true.copy(),
            label^,
            color.copy(),
            alpha,
            marker.copy(),
            linestyle.copy(),
        )
        var suffix = String("")
        if self.pos_label:
            suffix = " (Positive class: " + self._pos_label_text + ")"
        var next = ax.copy()
        next.reference_line = next.reference_line or ref_line
        next.xlabel = "Mean predicted probability" + suffix
        next.ylabel = "Fraction of positives" + suffix
        next.lines.append(line.copy())
        self.line_ = line^
        self.ax_ = next.copy()
        self.figure_ = CalibrationFigure(next.copy())
        ax = next^

    @staticmethod
    def from_predictions(
        y_true: Array,
        y_prob: Array,
        *,
        n_bins: Variant[Int, String] = 5,
        strategy: String = "uniform",
        pos_label: Optional[Float64] = None,
        name: Optional[String] = None,
        ref_line: Bool = True,
        color: String = "#2563eb",
        alpha: Float64 = 1,
        marker: String = "s",
        linestyle: String = "-",
    ) raises -> Self:
        vector(y_true)
        for i in range(y_true.size):
            if abs(y_true.item(i)) > 9007199254740991.0 or y_true.item(
                i
            ) != Float64(Int(y_true.item(i))):
                raise Error(
                    "ValueError: CalibrationDisplay requires binary"
                    " classification labels"
                )
        var curve = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
        )
        var positive = pos_label.value() if pos_label else Float64(1)
        var estimator_name = name.copy() if name else Optional[String](
            "Classifier"
        )
        var result = Self(
            curve[0],
            curve[1],
            y_prob,
            estimator_name=estimator_name,
            pos_label=positive,
        )
        if not pos_label:
            result._pos_label_text = "1"
        result.plot(
            ref_line=ref_line,
            color=color,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )
        return result^

    @staticmethod
    def from_estimator[
        E: CalibrationDisplayEstimator
    ](
        estimator: E,
        X: Array,
        y: Array,
        *,
        n_bins: Variant[Int, String] = 5,
        strategy: String = "uniform",
        pos_label: Optional[Float64] = None,
        name: Optional[String] = None,
        ref_line: Bool = True,
        color: String = "#2563eb",
        alpha: Float64 = 1,
        marker: String = "s",
        linestyle: String = "-",
    ) raises -> Self:
        var classes = estimator.class_labels()
        vector(classes)
        if classes.size != 2:
            raise Error(
                "ValueError: CalibrationDisplay requires a binary classifier"
            )
        var positive = pos_label.value() if pos_label else classes.item(1)
        var column = -1
        for j in range(2):
            if classes.item(j) == positive:
                column = j
        if column < 0:
            raise Error(
                "ValueError: pos_label is absent from estimator classes"
            )
        var probabilities = estimator.calibration_probabilities(X)
        matrix(probabilities)
        if probabilities.shape[0] != X.shape[0] or probabilities.shape[1] != 2:
            raise Error(
                "ValueError: binary classifier must return two probability"
                " columns"
            )
        var y_prob = nm.zeros[DType.float64](nm.Shape(X.shape[0]))
        for i in range(X.shape[0]):
            y_prob[nm.Item(i)] = probabilities.item(i, column)
        var estimator_name = name.copy() if name else Optional[String](
            estimator.display_name()
        )
        return Self.from_predictions(
            y,
            y_prob,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=positive,
            name=estimator_name,
            ref_line=ref_line,
            color=color,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )

    @staticmethod
    def from_predictions(
        y_true: Array,
        y_prob: Array,
        *,
        mut ax: CalibrationAxes,
        n_bins: Variant[Int, String] = 5,
        strategy: String = "uniform",
        pos_label: Optional[Float64] = None,
        name: Optional[String] = None,
        ref_line: Bool = True,
        color: String = "#2563eb",
        alpha: Float64 = 1,
        marker: String = "s",
        linestyle: String = "-",
    ) raises -> Self:
        var result = Self.from_predictions(
            y_true,
            y_prob,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
            name=name,
            ref_line=False,
        )
        result.plot(
            ax=ax,
            ref_line=ref_line,
            color=color,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )
        return result^

    @staticmethod
    def from_estimator[
        E: CalibrationDisplayEstimator
    ](
        estimator: E,
        X: Array,
        y: Array,
        *,
        mut ax: CalibrationAxes,
        n_bins: Variant[Int, String] = 5,
        strategy: String = "uniform",
        pos_label: Optional[Float64] = None,
        name: Optional[String] = None,
        ref_line: Bool = True,
        color: String = "#2563eb",
        alpha: Float64 = 1,
        marker: String = "s",
        linestyle: String = "-",
    ) raises -> Self:
        var result = Self.from_estimator(
            estimator,
            X,
            y,
            n_bins=n_bins,
            strategy=strategy,
            pos_label=pos_label,
            name=name,
            ref_line=False,
        )
        result.plot(
            ax=ax,
            ref_line=ref_line,
            color=color,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )
        return result^
