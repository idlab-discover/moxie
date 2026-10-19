# Mapping adapted from Treelite 4.6.1 src/model_loader/sklearn.cc
# (Apache-2.0, see LICENSES/Treelite.txt).
"""Treelite v4 export of fitted dense trees through the pinned Balsa codec.

The mapping reproduces Treelite 4.6.1 `treelite.sklearn.import_model` for a
one-tree RandomForest/ExtraTrees ensemble (src/model_loader/sklearn.cc), so a
saved checkpoint is byte-identical to the reference. Like that reference, every
split stores default_left=True: NaN inputs follow the left child in Treelite
even where the fitted tree's missing_go_to_left routes them right.
"""
from balsa import Model, ModelBuilder, TreeBuilder, Operator, TaskType
from moxie.tree._classes import BaseDecisionTree
from moxie.tree._records import TREE_LEAF, SPLIT_NUMERIC
from moxie.tree._tree import _go_left
from std.builtin.sort import sort

comptime TreeliteModel = Model[DType.float64]


def export_treelite[classifier: Bool, extra: Bool, label_dtype: DType, string_labels: Bool, //](tree: BaseDecisionTree[classifier, extra, label_dtype, string_labels]) raises -> TreeliteModel:
    """Convert a fitted tree to a validated float64 Treelite model.

    Save it with `balsa.save(model, path)`. Raises for unfitted trees,
    categorical splits and classifier outputs with fewer than two classes,
    which the Treelite reference also rejects.
    """
    if not tree.__sklearn_is_fitted__():
        raise Error("NotFittedError: call fit before export_treelite")
    ref data = tree.tree_
    var n_targets = tree.n_outputs_
    var max_classes = data.max_n_classes
    var stride = n_targets * max_classes
    var num_class = List[Int32]()
    for t in range(n_targets):
        comptime if classifier:
            if tree.n_classes_[t] < 2:
                raise Error("ValueError: Treelite requires at least 2 classes per output; output " + String(t) + " has " + String(tree.n_classes_[t]))
            num_class.append(Int32(tree.n_classes_[t]))
        else:
            num_class.append(1)
    var builder = ModelBuilder[DType.float64](
        Int32(tree.n_features_in_),
        task_type=TaskType.MULTI_CLF if classifier else TaskType.REGRESSOR,
        average_tree_output=True,
        num_class=num_class^,
        leaf_vector_shape=[Int32(n_targets), Int32(max_classes)],
        postprocessor="identity_multiclass" if classifier else "identity",
        attributes="{}",
    )

    var n_nodes = data.node_count
    var nodes = TreeBuilder[DType.float64](n_nodes)
    var total = Float64(data.nodes[0].n_node_samples)
    var gains = List[Float64]()
    for i in range(n_nodes):
        ref node = data.nodes[i]
        if node.left_child == TREE_LEAF:
            gains.append(0)
            var base = i * stride
            comptime if classifier:
                var leaf = List[Float64]()
                for j in range(stride):
                    leaf.append(data.value.unsafe_get(base + j))
                for t in range(n_targets):
                    var norm: Float64 = 0
                    for c in range(tree.n_classes_[t]):
                        norm += leaf[t * max_classes + c]
                    for c in range(tree.n_classes_[t]):
                        leaf[t * max_classes + c] /= norm
                nodes.leaf_vector(i, leaf)
            else:
                if n_targets == 1:
                    nodes.leaf(i, data.value.unsafe_get(base))
                else:
                    var leaf = List[Float64]()
                    for t in range(n_targets):
                        leaf.append(data.value.unsafe_get(base + t))
                    nodes.leaf_vector(i, leaf)
        else:
            if node.split_kind != SPLIT_NUMERIC:
                raise Error("ValueError: export_treelite supports numerical splits only; node " + String(i) + " is categorical")
            ref left = data.nodes[node.left_child]
            ref right = data.nodes[node.right_child]
            # Same operation order as Treelite's LoadSKLearnModel gain.
            var n = Float64(node.n_node_samples)
            gains.append(n * (node.impurity - Float64(left.n_node_samples) * left.impurity / n - Float64(right.n_node_samples) * right.impurity / n) / total)
            nodes.numerical_split(i, feature=Int32(node.feature), threshold=node.threshold, left=Int32(node.left_child), right=Int32(node.right_child), op=Operator.LE, default_left=True)

    var result = nodes^.build()
    # Treelite keeps an optional statistic empty until a node sets it; gain is
    # set only on splits, so a single-leaf tree has no gain arrays.
    var has_split = data.nodes[0].left_child != TREE_LEAF
    # Nodes without a leaf vector hold empty ranges at the running vector length.
    var offset: UInt64 = 0
    for i in range(n_nodes):
        if result.leaf_vector_end[i] > result.leaf_vector_begin[i]:
            offset = result.leaf_vector_end[i]
        else:
            result.leaf_vector_begin[i] = offset
            result.leaf_vector_end[i] = offset
    for i in range(n_nodes):
        ref node = data.nodes[i]
        result.data_count.append(UInt64(node.n_node_samples))
        result.data_count_present.append(1)
        result.sum_hess.append(node.weighted_n_node_samples)
        result.sum_hess_present.append(1)
        if has_split:
            result.gain.append(gains[i])
            result.gain_present.append(UInt8(0 if node.left_child == TREE_LEAF else 1))
    var target_id: Int32 = -1 if classifier or n_targets > 1 else 0
    var class_id: Int32 = -1 if classifier else 0
    builder.add_tree(result^, target_id=target_id, class_id=class_id)
    return builder^.build()


# Text layout adapted from sklearn/tree/_export.py (BSD-3-Clause).
# Copyright the scikit-learn developers; see LICENSES/scikit-learn.txt.
def _fixed(value: Float64, decimals: Int) raises -> String:
    """Exact binary64 to fixed decimal, rounded to nearest with ties to even.

    String.format lacks precision support on Mojo 1.0; NuMojo's formatter
    scales through floats and Int, losing digits and overflowing large values.
    Decimal digits here are text-conversion workspace, not estimator buffers.
    """
    from std.memory import bitcast
    var bits = bitcast[DType.uint64](value)
    var negative = (bits >> 63) != 0
    var exponent = Int((bits >> 52) & 2047)
    var mantissa = Int(bits & 0x000fffffffffffff)
    if exponent == 2047:
        if mantissa != 0:
            return "nan"
        return "-inf" if negative else "inf"
    if exponent != 0:
        mantissa += 4503599627370496
    var power = (exponent - 1023 - 52) if exponent != 0 else -1074
    var digits = List[Int]()
    while mantissa != 0:
        digits.append(mantissa % 10)
        mantissa //= 10
    if len(digits) == 0:
        digits.append(0)
    var scale = max(0, -power)
    for _ in range(abs(power)):
        var carry = 0
        for i in range(len(digits)):
            var product = digits[i] * (2 if power >= 0 else 5) + carry
            digits[i] = product % 10
            carry = product // 10
        if carry != 0:
            digits.append(carry)
    while len(digits) <= scale:
        digits.append(0)
    if decimals < scale:
        var cut = scale - decimals
        var sticky = False
        for i in range(cut - 1):
            sticky = sticky or digits[i] != 0
        var up = digits[cut - 1] > 5 or (digits[cut - 1] == 5 and (sticky or digits[cut] % 2 != 0))
        var rounded = List[Int]()
        var carry = Int(up)
        for i in range(cut, len(digits)):
            var digit = digits[i] + carry
            rounded.append(digit % 10)
            carry = digit // 10
        if carry:
            rounded.append(carry)
        digits = rounded^
        scale = decimals
    var result = String("-" if negative else "")
    for i in range(len(digits) - 1, scale - 1, -1):
        result += String(digits[i])
    if decimals > 0:
        result += "."
        for i in range(scale - 1, -1, -1):
            result += String(digits[i])
        for _ in range(decimals - scale):
            result += "0"
    return result


def export_text[classifier: Bool, extra: Bool, label_dtype: DType, string_labels: Bool, //](decision_tree: BaseDecisionTree[classifier, extra, label_dtype, string_labels], *, feature_names: Optional[List[String]] = None, class_names: Optional[List[String]] = None, max_depth: Int = 10, spacing: Int = 3, decimals: Int = 2, show_weights: Bool = False) raises -> String:
    """Render tree rules. Categorical rules name both sets and missing routes."""
    from moxie.tree._tree import _go_left
    decision_tree._ready()
    if max_depth < 0 or spacing < 1 or decimals < 0:
        raise Error("ValueError: max_depth/decimals must be nonnegative and spacing positive")
    if feature_names and len(feature_names.value()) != decision_tree.n_features_in_:
        raise Error("ValueError: feature_names must match the fitted feature count")
    comptime if classifier:
        if class_names:
            if decision_tree.n_outputs_ != 1:
                raise Error("ValueError: class_names is unsupported for multi-output classification")
            if len(class_names.value()) != decision_tree.n_classes_[0]:
                raise Error("ValueError: class_names must match the fitted class count")
    ref data = decision_tree.tree_
    # Postorder heights and rendering stacks avoid recursion on deep trees.
    var heights = List[Int]()
    for _ in range(data.node_count):
        heights.append(1)
    for i in range(data.node_count - 1, -1, -1):
        ref node = data.nodes[i]
        if node.left_child != TREE_LEAF:
            heights[i] = 1 + max(heights[node.left_child], heights[node.right_child])
    var nodes: List[Int] = [0]
    var depths: List[Int] = [1]
    var edges: List[String] = [String("")]
    var report = String("")
    while len(nodes) != 0:
        var node_id = nodes.pop()
        var depth = depths.pop()
        report += edges.pop()
        ref node = data.nodes[node_id]
        var indent = String("")
        for _ in range(depth - 1):
            indent += "|" + " " * spacing
        indent += "|" + "-" * spacing
        if node.left_child == TREE_LEAF:
            var values = List[Float64]()
            var base = node_id * data.n_outputs * data.max_n_classes
            if data.n_outputs == 1:
                for c in range(data.n_classes[0]):
                    values.append(data.value.unsafe_get(base + c))
            else:
                for k in range(data.n_outputs):
                    values.append(data.value.unsafe_get(base + k * data.max_n_classes))
            var formatted = List[String]()
            comptime if classifier:
                var chosen = 0
                for c in range(len(values)):
                    if values[c] > values[chosen]:
                        chosen = c
                    if show_weights:
                        formatted.append(_fixed(values[c] * node.weighted_n_node_samples, decimals))
                var label = String(chosen)
                if data.n_outputs == 1 and data.n_classes[0] != 1:
                    label = class_names.value()[chosen] if class_names else String(decision_tree._class_label(0, chosen))
                report += indent
                if show_weights:
                    report += " weights: [" + ", ".join(formatted) + "]"
                report += " class: " + label + "\n"
            else:
                for value in values:
                    formatted.append(_fixed(value, decimals))
                report += indent + " value: [" + ", ".join(formatted) + "]\n"
        elif depth - 1 > max_depth:
            report += indent + " truncated branch of depth " + String(heights[node_id]) + "\n"
        else:
            var name = feature_names.value()[node.feature] if feature_names else "feature_" + String(node.feature)
            var left: String
            var right: String
            if node.split_kind == SPLIT_NUMERIC:
                var threshold = _fixed(node.threshold, decimals)
                left = name + " <= " + threshold
                right = name + " >  " + threshold
            else:
                var left_values = List[String]()
                var right_values = List[String]()
                for c in range(data.n_categories[node.feature]):
                    var label: String
                    if len(decision_tree._encoder.string_domains) != 0 and decision_tree._encoder.string_domains[node.feature]:
                        label = "{!r}".format(decision_tree._encoder.string_domains[node.feature].value().get(c))
                    else:
                        label = String(decision_tree._encoder.numeric_domains[node.feature].get(c))
                    if _go_left(node, Float32(c), data.n_categories[node.feature]):
                        left_values.append(label)
                    else:
                        right_values.append(label)
                left = name + " in {" + ", ".join(left_values) + "}"
                right = name + " in {" + ", ".join(right_values) + "}"
                if node.missing_go_to_left:
                    left += " or missing/unknown"
                else:
                    right += " or missing/unknown"
            nodes.append(node.right_child)
            depths.append(depth + 1)
            edges.append(indent + " " + right + "\n")
            nodes.append(node.left_child)
            depths.append(depth + 1)
            edges.append(indent + " " + left + "\n")
    return report


def _dot_number(value: Float64, precision: Int) raises -> String:
    """Fixed decimal adaptation: retain one fractional digit, avoid exponents."""
    var text = _fixed(value, precision)
    if "." in text:
        while text.endswith("0") and not text.endswith(".0"):
            var trimmed = String(text.removesuffix("0"))
            text = trimmed^
    elif text != "nan" and text != "inf" and text != "-inf":
        text += ".0"
    return text


def _dot_escape(text: String) -> String:
    return text.replace('\\', '\\\\').replace('"', '\\"')


def _node_lines[classifier: Bool, extra: Bool, label_dtype: DType, string_labels: Bool, //](
    decision_tree: BaseDecisionTree[classifier, extra, label_dtype, string_labels], node_id: Int,
    feature_names: Optional[List[String]], class_names: Optional[List[String]],
    label: String, impurity: Bool, node_ids: Bool, proportion: Bool, precision: Int,
) raises -> List[String]:
    from std.math import floor, round
    ref data = decision_tree.tree_
    ref node = data.nodes[node_id]
    var scale = Float64(10) ** precision
    var lines = List[String]()
    var labels = label == "all" or (label == "root" and node_id == 0)
    if node_ids:
        lines.append(("node " if labels else "") + "#" + String(node_id))
    if node.left_child != TREE_LEAF:
        var name = feature_names.value()[node.feature] if feature_names else "x[" + String(node.feature) + "]"
        if node.split_kind == SPLIT_NUMERIC:
            lines.append(name + " <= " + _dot_number(node.threshold, precision))
        else:
            var members = List[String]()
            for c in range(data.n_categories[node.feature]):
                if _go_left(node, Float32(c), data.n_categories[node.feature]):
                    if len(decision_tree._encoder.string_domains) != 0 and decision_tree._encoder.string_domains[node.feature]:
                        members.append("{!r}".format(decision_tree._encoder.string_domains[node.feature].value().get(c)))
                    else:
                        members.append(String(decision_tree._encoder.numeric_domains[node.feature].get(c)))
            lines.append(name + " in {" + ", ".join(members) + "}")
            lines.append("missing/unknown -> " + ("True" if node.missing_go_to_left else "False"))
    if impurity:
        lines.append((decision_tree.params.criterion + " = " if labels else "") + _dot_number(node.impurity, precision))
    var samples = String(node.n_node_samples)
    if proportion:
        samples = _dot_number(100.0 * Float64(node.n_node_samples) / Float64(data.nodes[0].n_node_samples), 1) + "%"
    lines.append(("samples = " if labels else "") + samples)
    var values = List[Float64]()
    var integral = data.n_classes[0] != 1 and not proportion
    for k in range(data.n_outputs):
        for c in range(data.max_n_classes):
            var value = data.value.unsafe_get((node_id * data.n_outputs + k) * data.max_n_classes + c)
            if not proportion and data.n_classes[0] != 1:
                value *= node.weighted_n_node_samples
            values.append(value)
            integral = integral and value == floor(value)
    var rows = List[String]()
    for k in range(data.n_outputs):
        var cells = List[String]()
        for c in range(data.max_n_classes):
            var value = values[k * data.max_n_classes + c]
            cells.append(_fixed(value, 0) if integral else _dot_number(round(value * scale) / scale, precision))
        var row = ", ".join(cells)
        if data.n_outputs != 1 or data.n_classes[0] != 1:
            row = "[" + row + "]"
        rows.append(row)
    var value_text = "\n".join(rows)
    if data.n_outputs != 1:
        value_text = "[" + value_text + "]"
    lines.append(("value = " if labels else "") + value_text)
    if class_names and data.n_classes[0] != 1 and data.n_outputs == 1:
        var chosen = 0
        for c in range(1, data.n_classes[0]):
            if values[c] > values[chosen]:
                chosen = c
        lines.append(("class = " if labels else "") + class_names.value()[chosen])
    return lines^


def export_graphviz[classifier: Bool, extra: Bool, label_dtype: DType, string_labels: Bool, //](
    decision_tree: BaseDecisionTree[classifier, extra, label_dtype, string_labels],
    *, max_depth: Optional[Int] = None,
    feature_names: Optional[List[String]] = None,
    class_names: Optional[List[String]] = None,
    label: String = "all", filled: Bool = False, leaves_parallel: Bool = False,
    impurity: Bool = True, node_ids: Bool = False, proportion: Bool = False,
    rotate: Bool = False, rounded: Bool = False, special_characters: Bool = False,
    precision: Int = 3, fontname: String = "helvetica",
    fill_colors: Optional[List[String]] = None,
) raises -> String:
    """Return DOT for numeric and categorical trees. Write the returned String with native I/O."""
    from std.math import floor, round
    decision_tree._ready()
    if precision < 0 or (max_depth and max_depth.value() < 0):
        raise Error("ValueError: precision and max_depth must be nonnegative")
    if label != "all" and label != "root" and label != "none":
        raise Error("ValueError: label must be all, root or none")
    if feature_names and len(feature_names.value()) != decision_tree.n_features_in_:
        raise Error("ValueError: feature_names must match the fitted feature count")
    ref data = decision_tree.tree_
    if class_names and data.n_outputs == 1 and data.n_classes[0] != 1 and len(class_names.value()) != data.n_classes[0]:
        raise Error("ValueError: class_names must match the fitted class count")
    var report = String("digraph Tree {\nnode [shape=box")
    if filled or rounded:
        report += ', style="' + ("filled, rounded" if filled and rounded else "filled" if filled else "rounded") + '", color="black"'
    report += ', fontname="' + _dot_escape(fontname) + '"] ;\n'
    if leaves_parallel:
        report += "graph [ranksep=equally, splines=polyline] ;\n"
    report += 'edge [fontname="' + _dot_escape(fontname) + '"] ;\n'
    if rotate:
        report += "rankdir=LR ;\n"
    var colors = _tree_colors(decision_tree, fill_colors) if filled else List[String]()
    var ranks = Dict[String, List[String]]()
    ranks["leaves"] = List[String]()
    var nodes: List[Int] = [0]
    var depths: List[Int] = [0]
    var parents: List[Int] = [-1]
    while nodes:
        var node_id = nodes.pop()
        var depth = depths.pop()
        var parent = parents.pop()
        ref node = data.nodes[node_id]
        var truncated = Bool(max_depth) and depth > max_depth.value()
        var lines = List[String]()
        if truncated:
            lines.append("(...)")
        else:
            lines = _node_lines(decision_tree, node_id, feature_names, class_names, label, impurity, node_ids, proportion, precision)
        var text = "\n".join(lines)
        if special_characters and not truncated:
            var escaped = List[String]()
            for i in range(len(lines)):
                var line = _xml_escape(lines[i])
                if i == 0 and node_ids:
                    line = line.replace("#", "&#35;")
                if i == Int(node_ids) and node.left_child != TREE_LEAF:
                    if node.split_kind == SPLIT_NUMERIC:
                        var name = _xml_escape(feature_names.value()[node.feature]) if feature_names else "x<SUB>" + String(node.feature) + "</SUB>"
                        line = name + " &le; " + _dot_number(node.threshold, precision)
                    elif not feature_names:
                        line = line.replace("x[" + String(node.feature) + "]", "x<SUB>" + String(node.feature) + "</SUB>")
                escaped.append(line)
            text = "<" + "<br/>".join(escaped).replace("\n", "<br/>") + ">"
        else:
            text = '"' + _dot_escape(text).replace("\n", "\\n") + '"'
        report += String(node_id) + " [label=" + text
        if filled:
            report += ', fillcolor="' + ("#C0C0C0" if truncated else colors[node_id]) + '"'
        report += "] ;\n"
        if leaves_parallel:
            var rank = "leaves" if truncated or node.left_child == TREE_LEAF else String(depth)
            if rank not in ranks:
                ranks[rank] = List[String]()
            ranks[rank].append(String(node_id))
        if parent != -1:
            report += String(parent) + " -> " + String(node_id)
            if parent == 0 and not truncated:
                var angle = (45 if node_id == 1 else -45) * (-1 if rotate else 1)
                report += ' [labeldistance=2.5, labelangle=' + String(angle) + ', headlabel="' + ("True" if node_id == 1 else "False") + '"]'
            report += " ;\n"
        if not truncated and node.left_child != TREE_LEAF:
            nodes.append(node.right_child)
            depths.append(depth + 1)
            parents.append(node_id)
            nodes.append(node.left_child)
            depths.append(depth + 1)
            parents.append(node_id)
    if leaves_parallel:
        var keys = List[String]()
        for key in ranks:
            keys.append(key)
        sort(keys)
        for key in keys:
            report += "{rank=same ; " + "; ".join(ranks[key]) + "} ;\n"
    return report + "}"


def _xml_escape(text: String) -> String:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")


def _tree_colors[classifier: Bool, extra: Bool, label_dtype: DType, string_labels: Bool, //](
    model: BaseDecisionTree[classifier, extra, label_dtype, string_labels],
    fill_colors: Optional[List[String]],
) raises -> List[String]:
    from std.math import floor, round, isfinite
    ref tree = model.tree_
    var count = tree.n_classes[0]
    var palette = List[SIMD[DType.int64, 4]]()
    var hex = String("0123456789abcdef")
    if fill_colors and len(fill_colors.value()) != count:
        raise Error("ValueError: fill_colors must match the first output class count")
    for i in range(count):
        var rgb = SIMD[DType.int64, 4](0)
        if fill_colors:
            var color = fill_colors.value()[i].lower()
            if color.byte_length() != 7 or not color.startswith("#"):
                raise Error("ValueError: fill_colors requires #RRGGBB strings")
            for c in range(3):
                for j in range(2):
                    var digit = String(color[byte=1 + c * 2 + j])
                    if digit not in hex:
                        raise Error("ValueError: fill_colors requires #RRGGBB strings")
                    rgb[c] = rgb[c] * 16 + Int64(hex.find(digit))
        else:
            var hue = Float64(Int(25.0 + Float64(i) * (360.0 / Float64(count)))) / 60.0
            var chroma: Float64 = 0.75 * 0.9
            var shift: Float64 = 0.9 - chroma
            var x = chroma * (1 - abs((hue - floor(hue / 2) * 2) - 1))
            var sector = Int(hue) % 6
            var channels = SIMD[DType.float64, 4](0)
            if sector == 0:
                channels[0] = chroma
                channels[1] = x
            elif sector == 1:
                channels[0] = x
                channels[1] = chroma
            elif sector == 2:
                channels[1] = chroma
                channels[2] = x
            elif sector == 3:
                channels[1] = x
                channels[2] = chroma
            elif sector == 4:
                channels[0] = x
                channels[2] = chroma
            else:
                channels[0] = chroma
                channels[2] = x
            for c in range(3):
                rgb[c] = Int64(255 * (channels[c] + shift))
        palette.append(rgb)
    var low = -tree.nodes[0].impurity if tree.n_outputs != 1 else tree.value.unsafe_get(0)
    var high = low
    for i in range(tree.node_count):
        var value = -tree.nodes[i].impurity if tree.n_outputs != 1 else tree.value.unsafe_get(i * tree.max_n_classes)
        low = min(low, value)
        high = max(high, value)
    var bounded = tree.n_outputs != 1 or (count == 1 and low != high)
    var colors = List[String]()
    for i in range(tree.node_count):
        var chosen = 0
        var alpha: Float64 = 0
        if bounded:
            var value = -tree.nodes[i].impurity if tree.n_outputs != 1 else tree.value.unsafe_get(i * tree.max_n_classes)
            # A constant multioutput tree has no color range; render it white.
            if high != low:
                alpha = (value - low) / (high - low)
        elif count > 1:
            var base = i * count
            var first = tree.value.unsafe_get(base)
            var second = Float64("-inf")
            for c in range(1, count):
                var value = tree.value.unsafe_get(base + c)
                if value > first:
                    second = first
                    first = value
                    chosen = c
                else:
                    second = max(second, value)
            alpha = (first - second) / (1 - second)
        if not isfinite(alpha) or alpha < 0 or alpha > 1:
            raise Error("ValueError: tree statistics cannot be represented as fill colors")
        var color = String("#")
        for c in range(3):
            var channel = Int(round(alpha * Float64(palette[chosen][c]) + (1 - alpha) * 255))
            color += String(hex[byte=channel // 16]) + String(hex[byte=channel % 16])
        colors.append(color)
    return colors^


@fieldwise_init
struct _PlotNode(Copyable, Movable):
    var node_id: Int
    var parent: Int
    var depth: Int
    var left: Int
    var right: Int
    var x: Float64
    var text: List[String]
    var color: String


def plot_tree[classifier: Bool, extra: Bool, label_dtype: DType, string_labels: Bool, //](
    decision_tree: BaseDecisionTree[classifier, extra, label_dtype, string_labels],
    *, max_depth: Optional[Int] = None,
    feature_names: Optional[List[String]] = None,
    class_names: Optional[List[String]] = None,
    label: String = "all", filled: Bool = False, impurity: Bool = True,
    node_ids: Bool = False, proportion: Bool = False, rounded: Bool = False,
    precision: Int = 3, fontsize: Optional[Float64] = None,
    fill_colors: Optional[List[String]] = None,
) raises -> String:
    """Return a standalone SVG. Native layout uses leaf slots, not Matplotlib axes."""
    from std.math import isfinite
    decision_tree._ready()
    if precision < 0 or (max_depth and max_depth.value() < 0):
        raise Error("ValueError: precision and max_depth must be nonnegative")
    if label != "all" and label != "root" and label != "none":
        raise Error("ValueError: label must be all, root or none")
    if feature_names and len(feature_names.value()) != decision_tree.n_features_in_:
        raise Error("ValueError: feature_names must match the fitted feature count")
    ref data = decision_tree.tree_
    if class_names and data.n_outputs == 1 and data.n_classes[0] != 1 and len(class_names.value()) != data.n_classes[0]:
        raise Error("ValueError: class_names must match the fitted class count")
    var size = fontsize.value() if fontsize else Float64(14)
    if not isfinite(size) or size <= 0:
        raise Error("ValueError: fontsize must be finite and positive")
    var colors = _tree_colors(decision_tree, fill_colors) if filled else List[String]()
    var layout = List[_PlotNode]()
    var nodes: List[Int] = [0]
    var parents: List[Int] = [-1]
    var depths: List[Int] = [0]
    var leaves = 0
    var deepest = 0
    var longest = 0
    var most_lines = 0
    while nodes:
        var node_id = nodes.pop()
        var parent = parents.pop()
        var depth = depths.pop()
        var truncated = Bool(max_depth) and depth > max_depth.value()
        ref node = data.nodes[node_id]
        var lines = List[String]()
        if truncated:
            lines.append("(...)")
        else:
            var content = _node_lines(decision_tree, node_id, feature_names, class_names, label, impurity, node_ids, proportion, precision)
            for line in content:
                for part in line.split("\n"):
                    lines.append(String(part))
        for line in lines:
            longest = max(longest, line.byte_length())
        most_lines = max(most_lines, len(lines))
        deepest = max(deepest, depth)
        var index = len(layout)
        var color = "#c0c0c0" if truncated else colors[node_id] if filled else "#ffffff"
        layout.append(_PlotNode(node_id, parent, depth, -1, -1, 0, lines^, color))
        if parent >= 0:
            if layout[parent].left == -1:
                layout[parent].left = index
            else:
                layout[parent].right = index
        if truncated or node.left_child == TREE_LEAF:
            layout[index].x = Float64(leaves)
            leaves += 1
        else:
            nodes.append(node.right_child)
            parents.append(index)
            depths.append(depth + 1)
            nodes.append(node.left_child)
            parents.append(index)
            depths.append(depth + 1)
    for i in range(len(layout) - 1, -1, -1):
        if layout[i].left != -1:
            layout[i].x = (layout[layout[i].left].x + layout[layout[i].right].x) / 2
    var box_width = Float64(longest + 4) * size
    var line_height = size * 1.4
    var box_height = Float64(most_lines + 1) * line_height
    var gap = size * 4
    var width = Float64(leaves) * (box_width + gap)
    var height = Float64(deepest + 1) * (box_height + gap)
    if not isfinite(width) or not isfinite(height):
        raise Error("ValueError: fontsize produces nonfinite SVG dimensions")
    var svg = String('<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Decision tree" viewBox="0 0 ')
    svg += String(width) + " " + String(height) + '">\n<title>Decision tree</title>\n'
    svg += '<rect width="100%" height="100%" fill="#ffffff"/>\n'
    svg += '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker></defs>\n'
    for i in range(len(layout)):
        ref item = layout[i]
        var x = (item.x + 0.5) * (box_width + gap)
        var y = Float64(item.depth) * (box_height + gap) + gap / 2
        if item.parent >= 0:
            ref parent = layout[item.parent]
            var px = (parent.x + 0.5) * (box_width + gap)
            var py = Float64(parent.depth) * (box_height + gap) + gap / 2 + box_height
            svg += '<path data-parent="' + String(parent.node_id) + '" data-child="' + String(item.node_id) + '" d="M ' + String(px) + ' ' + String(py) + ' L ' + String(x) + ' ' + String(y) + '" fill="none" stroke="#555" marker-end="url(#arrow)"/>\n'
            if item.parent == 0:
                svg += '<text x="' + String((px + x) / 2) + '" y="' + String((py + y) / 2 - size / 3) + '" text-anchor="middle" font-family="monospace" font-size="' + String(size) + '">' + ("True" if parent.left == i else "False") + '</text>\n'
        svg += '<g data-node-id="' + String(item.node_id) + '">\n<rect x="' + String(x - box_width / 2) + '" y="' + String(y) + '" width="' + String(box_width) + '" height="' + String(box_height) + '" rx="' + String(size if rounded else Float64(0)) + '" fill="' + item.color + '" stroke="#333"/>\n'
        for j in range(len(item.text)):
            svg += '<text x="' + String(x) + '" y="' + String(y + Float64(j + 1) * line_height) + '" text-anchor="middle" font-family="monospace" font-size="' + String(size) + '">' + _xml_escape(item.text[j]) + '</text>\n'
        svg += '</g>\n'
    return svg + '</svg>\n'
