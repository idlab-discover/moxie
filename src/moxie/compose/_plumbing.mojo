"""Dense composition metadata and extension contracts, not estimators."""
from moxie.base import BaseEstimator, clone
from moxie.pipeline import PipelineEstimator
from numojo import NDArray


struct ColumnSelection(Copyable, Movable):
    """Resolved positional selection; scalar selection preserves rank one."""
    var indices: List[Int]
    var scalar: Bool

    def __init__(out self, indices: List[Int], n_features: Int, scalar: Bool = False) raises:
        if n_features < 0 or (scalar and len(indices) != 1):
            raise Error("ValueError: invalid selection shape")
        self.indices = List[Int]()
        self.scalar = scalar
        for index in indices:
            var resolved = index
            if resolved < 0:
                resolved += n_features
            if resolved < 0 or resolved >= n_features:
                raise Error("ValueError: column index out of bounds")
            self.indices.append(resolved)

    def remainder(self, n_features: Int) raises -> Self:
        var remaining = List[Int]()
        for i in range(n_features):
            var used = False
            for index in self.indices:
                used = used or index == i
            if not used:
                remaining.append(i)
        # Validate stored indices against the requested schema, too.
        var checked = Self(self.indices, n_features)
        return Self(remaining, n_features)


@fieldwise_init
struct OutputSpan(Copyable, Movable):
    var start: Int
    var stop: Int


struct BlockLayout(Copyable, Movable):
    """Output slices in declaration order; skipped blocks use [0, 0)."""
    var n_rows: Int
    var width: Int
    var spans: List[OutputSpan]

    def __init__(out self, n_rows: Int) raises:
        if n_rows < 0:
            raise Error("ValueError: negative sample count")
        self.n_rows = n_rows
        self.width = 0
        self.spans = List[OutputSpan]()

    def append(mut self, n_rows: Int, width: Int, active: Bool = True) raises:
        if n_rows != self.n_rows or width < 0 or (not active and width != 0):
            raise Error("ValueError: incompatible output block")
        if active:
            self.spans.append(OutputSpan(self.width, self.width + width))
            self.width += width
        else:
            self.spans.append(OutputSpan(0, 0))


def feature_name(transformer: String, feature: String, verbose: Bool = True) -> String:
    if verbose:
        return transformer + "__" + feature
    return feature


def validate_feature_names(names: List[String]) raises:
    for i in range(len(names)):
        for j in range(i):
            if names[i] == names[j]:
                raise Error("ValueError: output feature names are not unique")


def validate_target_mapping(has_transformer: Bool, has_func: Bool, has_inverse: Bool) raises:
    if has_transformer and (has_func or has_inverse):
        raise Error("ValueError: transformer and functions are mutually exclusive")
    if has_func != has_inverse:
        raise Error("ValueError: func and inverse_func must be provided together")


def clone_component[E: BaseEstimator](component: E) raises -> E:
    """Delegate nested ownership to the foundation clone contract."""
    return clone(component)


trait ColumnResolver(Movable):
    comptime Schema: Movable
    def resolve(self, schema: Self.Schema) raises -> ColumnSelection:
        ...


trait FeatureNamer(Movable):
    def name(self, transformer: String, feature: String) raises -> String:
        ...


trait DenseBlockAssembler(Movable):
    def concatenate[dtype: DType](self, blocks: List[NDArray[dtype]], layout: BlockLayout) raises -> NDArray[dtype]:
        ...


trait NestedComposition(PipelineEstimator):
    """Typed deep parameters/clone inherited from pipeline and foundation.

    Routing remains owned by pipeline; no dynamic kwargs registry here.
    """
    pass
