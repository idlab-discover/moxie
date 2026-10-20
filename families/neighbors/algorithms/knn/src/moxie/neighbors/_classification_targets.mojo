"""Neighbor classification adapters over the shared exact label domains."""
import numojo as nm
from std.math import isfinite
from moxie._labels import StringTarget, encode_column, type_of_target
from moxie.neighbors._supervised import WeightedNeighbors, probabilities, at

struct LabelOps[Y: Copyable & Deinitable]:
    comptime dtype = DType.int64 if Self.Y == nm.NDArray[DType.int64] else (DType.float32 if Self.Y == nm.NDArray[DType.float32] else DType.float64)

    @staticmethod
    def kind(y: Self.Y) raises -> String:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            return type_of_target(rebind[StringTarget](y))
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype]
            return type_of_target(rebind[nm.NDArray[Self.dtype]](y))

    @staticmethod
    def rows(y: Self.Y) raises -> Int:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            return rebind[StringTarget](y).rows()
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype], "labels require StringTarget or Int64/Float32/Float64 NDArray"
            return rebind[nm.NDArray[Self.dtype]](y).shape[0]

    @staticmethod
    def rank(y: Self.Y) -> Int:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            return rebind[StringTarget](y).rank()
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype]
            return rebind[nm.NDArray[Self.dtype]](y).ndim

    @staticmethod
    def columns(y: Self.Y) raises -> Int:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            return rebind[StringTarget](y).columns()
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype]
            ref a = rebind[nm.NDArray[Self.dtype]](y)
            return 1 if a.ndim == 1 else a.shape[1]

    @staticmethod
    def allocate(rows: Int, columns: Int, matrix: Bool) raises -> Self.Y:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            var values = List[String]()
            for _ in range(rows*columns): values.append("")
            var result = StringTarget(values,rows,columns) if matrix else StringTarget(values)
            return rebind[Self.Y](result).copy()
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype]
            var result = nm.zeros[Self.dtype](nm.Shape(rows,columns) if matrix else nm.Shape(rows))
            return rebind[Self.Y](result).copy()

    @staticmethod
    def copy_value(mut dest: Self.Y, i: Int, j: Int, source: Self.Y, row: Int, col: Int) raises:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            rebind[StringTarget](dest).set(i,rebind[StringTarget](source).get(row,col),j)
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype]
            ref a = rebind[nm.NDArray[Self.dtype]](source)
            ref b = rebind[nm.NDArray[Self.dtype]](dest)
            var src = row*Int(a.strides[0]) + (col*Int(a.strides[1]) if a.ndim == 2 else 0)
            var dst = i*Int(b.strides[0]) + (j*Int(b.strides[1]) if b.ndim == 2 else 0)
            b.unsafe_set(dst,a.unsafe_get(src))

    @staticmethod
    def equal(a: Self.Y, i: Int, j: Int, b: Self.Y, row: Int, col: Int) raises -> Bool:
        comptime if Self.Y == StringTarget:
            comptime assert Self.Y == StringTarget
            return rebind[StringTarget](a).get(i,j) == rebind[StringTarget](b).get(row,col)
        else:
            comptime assert Self.Y == nm.NDArray[Self.dtype]
            ref x = rebind[nm.NDArray[Self.dtype]](a)
            ref y = rebind[nm.NDArray[Self.dtype]](b)
            return x.unsafe_get(i*Int(x.strides[0])+(j*Int(x.strides[1]) if x.ndim == 2 else 0)) == y.unsafe_get(row*Int(y.strides[0])+(col*Int(y.strides[1]) if y.ndim == 2 else 0))

struct ClassificationTargets[Y: Copyable & Deinitable](Copyable, Movable):
    comptime Ops = LabelOps[Self.Y]
    var classes: List[Self.Y]
    var codes: nm.NDArray[DType.int64]
    var code_classes: List[nm.NDArray[DType.int64]]
    var outputs_2d: Bool

    def __init__(out self, y: Self.Y, n: Int) raises:
        if Self.Ops.rank(y) != 1 and Self.Ops.rank(y) != 2: raise Error("ValueError: target must have rank one or two")
        if Self.Ops.rows(y) != n or n == 0: raise Error("ValueError: target row mismatch")
        var cols = Self.Ops.columns(y)
        if cols == 0: raise Error("ValueError: empty target")
        self.outputs_2d = Self.Ops.rank(y) == 2 and cols != 1
        self.classes = List[Self.Y]()
        self.code_classes = List[nm.NDArray[DType.int64]]()
        self.codes = nm.zeros[DType.int64](nm.Shape(n,cols))
        for col in range(cols):
            var inverse: nm.NDArray[DType.int64]
            comptime if Self.Y == StringTarget:
                comptime assert Self.Y == StringTarget
                var encoded = encode_column(rebind[StringTarget](y),col)
                var domain = StringTarget(encoded.classes)
                self.classes.append(rebind[Self.Y](domain).copy())
                inverse = encoded.index.inverse.copy()
            else:
                comptime assert Self.Y == nm.NDArray[Self.Ops.dtype]
                var encoded = encode_column(rebind[nm.NDArray[Self.Ops.dtype]](y),col)
                self.classes.append(rebind[Self.Y](encoded.classes).copy())
                inverse = encoded.index.inverse.copy()
            var ids = nm.zeros[DType.int64](nm.Shape(Self.Ops.rows(self.classes[col])))
            for c in range(ids.shape[0]): ids.unsafe_set(c,Int64(c))
            self.code_classes.append(ids^)
            for i in range(n): self.codes.unsafe_set(i*cols+col,inverse.unsafe_get(i))

    def outlier_codes(self, outliers: Optional[Self.Y]) raises -> Optional[nm.NDArray[DType.int64]]:
        if not outliers: return None
        if Self.Ops.rank(outliers.value()) != 1 or Self.Ops.rows(outliers.value()) != len(self.classes):
            raise Error("ValueError: one outlier label required per output")
        var validated = ClassificationTargets(outliers.value(),len(self.classes))
        var result = nm.full[DType.int64](nm.Shape(len(self.classes)),Int64(-1))
        for col in range(len(self.classes)):
            for c in range(Self.Ops.rows(self.classes[col])):
                if Self.Ops.equal(outliers.value(),col,0,self.classes[col],c,0): result.unsafe_set(col,Int64(c))
        return result^

    def most_frequent(self) raises -> Self.Y:
        var result = Self.Ops.allocate(len(self.classes),1,False)
        for col in range(len(self.classes)):
            var best = 0
            var best_count = -1
            for c in range(Self.Ops.rows(self.classes[col])):
                var count = 0
                for i in range(self.codes.shape[0]):
                    if self.codes.unsafe_get(i*len(self.classes)+col) == Int64(c): count += 1
                if count > best_count:
                    best_count = count
                    best = c
            Self.Ops.copy_value(result,col,0,self.classes[col],best,0)
        return result^

    def proba(self, rows: WeightedNeighbors, outliers: Optional[Self.Y]) raises -> List[nm.NDArray[DType.float64]]:
        return probabilities(self.codes,self.code_classes,rows,self.outlier_codes(outliers))

    def predict(self, probs: List[nm.NDArray[DType.float64]], outliers: Optional[Self.Y]) raises -> Self.Y:
        var result = Self.Ops.allocate(probs[0].shape[0],len(self.classes),self.outputs_2d)
        for col in range(len(self.classes)):
            for i in range(probs[col].shape[0]):
                var best = 0
                var total = Float64(0)
                for c in range(probs[col].shape[1]):
                    total += at(probs[col],i,c)
                    if at(probs[col],i,c) > at(probs[col],i,best): best = c
                if total == 0 and outliers: Self.Ops.copy_value(result,i,col,outliers.value(),col,0)
                else: Self.Ops.copy_value(result,i,col,self.classes[col],best,0)
        return result^

    @staticmethod
    def score[dtype: DType](pred: Self.Y, y: Self.Y, sample_weight: Optional[nm.NDArray[dtype]]) raises -> Float64:
        var validated = ClassificationTargets(y,Self.Ops.rows(pred))
        if Self.Ops.kind(y) == "multiclass-multioutput": raise Error("ValueError: multiclass-multioutput is not supported by accuracy_score")
        if Self.Ops.columns(y) != Self.Ops.columns(pred): raise Error("ValueError: output count mismatch")
        var n = Self.Ops.rows(y)
        if sample_weight:
            if sample_weight.value().ndim != 1 or sample_weight.value().shape[0] != n: raise Error("ValueError: invalid sample_weight shape")
        var total = Float64(0)
        var denominator = Float64(0)
        for i in range(n):
            var weight = Float64(1)
            if sample_weight: weight = Float64(sample_weight.value().unsafe_get(i*Int(sample_weight.value().strides[0])))
            if not isfinite(weight) or weight < 0: raise Error("ValueError: invalid sample weight")
            var correct = True
            for col in range(Self.Ops.columns(y)):
                correct = correct and Self.Ops.equal(pred,i,col,y,i,col)
            if correct: total += weight
            denominator += weight
        if denominator == 0: raise Error("ValueError: sample weights sum to zero")
        return total/denominator
