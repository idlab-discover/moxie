"""Native scalar-search and unbounded NCA solver oracle bridge."""
import numojo as nm
from std.sys import argv
from moxie.neighbors._nca_linesearch import MoreThuente, dcstep
from moxie.neighbors._nca_lbfgs import NCAObjective, NCACallback, minimize
from moxie.neighbors._nca_math import Matrix


struct Trace(NCACallback):
    def __init__(out self):
        pass

    def __call__(mut self, x: Matrix, iteration: Int) raises:
        print("C", iteration, x.unsafe_get(0), x.unsafe_get(1))


@fieldwise_init
struct Objective(NCAObjective):
    var kind: Int

    def __call__(self, x: Matrix) raises -> Tuple[Float64, Matrix]:
        var a = x.unsafe_get(0)
        var b = x.unsafe_get(1)
        print("E", a, b)
        var g = nm.zeros[DType.float64](nm.Shape(1, 2))
        var f: Float64
        if self.kind == 0:
            f = (a * a + 10 * b * b) / 2
            g.unsafe_set(0, a)
            g.unsafe_set(1, 10 * b)
        elif self.kind == 1:
            f = (1 - a) ** 2 + 100 * (b - a * a) ** 2
            g.unsafe_set(0, 2 * (a - 1) - 400 * a * (b - a * a))
            g.unsafe_set(1, 200 * (b - a * a))
        elif self.kind == 2:
            f = -a
            g.unsafe_set(0, -1)
        elif self.kind == 3:
            f = a * a / 2
            g.unsafe_set(0, -a)  # Deliberately inconsistent: search must fail.
        elif self.kind == 4:
            f = (a ** 4 + b ** 4) / 4
            g.unsafe_set(0, a ** 3)
            g.unsafe_set(1, b ** 3)
        else:
            # First step is accepted; later inconsistent derivatives require
            # history reset and rollback without any extra callback.
            f = a * a / 2
            g.unsafe_set(0, a if a > 1.5 else -a)
        return (f, g^)


def main() raises:
    var args = List[String]()
    for arg in argv():
        args.append(String(arg))
    if args[1] == "solve":
        var x = nm.zeros[DType.float64](nm.Shape(1, 2))
        x.unsafe_set(0, Float64(args[3]))
        x.unsafe_set(1, Float64(args[4]))
        var trace = Trace()
        var result = minimize(Objective(Int(args[2])), x^, trace,
            Int(args[5]), Float64(args[6]), maxfun=Int(args[7]), maxls=Int(args[8]))
        print("R", result.iterations, result.evaluations, result.status,
              result.restarts, result.value, result.x.unsafe_get(0),
              result.x.unsafe_get(1))
        return
    var tokens = List[String]()
    with open(args[2], "r") as file:
        var contents = file.read()
        for token in contents.split():
            tokens.append(String(token))
    if args[1] == "step":
        for i in range(0, len(tokens), 12):
            var stx = Float64(tokens[i])
            var fx = Float64(tokens[i + 1])
            var dx = Float64(tokens[i + 2])
            var sty = Float64(tokens[i + 3])
            var fy = Float64(tokens[i + 4])
            var dy = Float64(tokens[i + 5])
            var stp = Float64(tokens[i + 6])
            var brackt = Bool(Int(tokens[i + 9]))
            dcstep(stx, fx, dx, sty, fy, dy, stp,
                Float64(tokens[i + 7]), Float64(tokens[i + 8]), brackt,
                Float64(tokens[i + 10]), Float64(tokens[i + 11]))
            print("D", stx, fx, dx, sty, fy, dy, stp, Int(brackt))
        return
    var search = MoreThuente()
    var step = Float64(tokens[0])
    for i in range(6, len(tokens), 2):
        search.search(Float64(tokens[i]), Float64(tokens[i + 1]), step,
            Float64(tokens[1]), Float64(tokens[2]), Float64(tokens[3]),
            Float64(tokens[4]), Float64(tokens[5]))
        print("S", step, search.task, search.stage, Int(search.brackt))
