import numojo as nm
from std.time import perf_counter_ns
from std.sys import argv
from moxie._nb_counts import MultinomialNB

def main() raises:
    var args = argv()
    var loops = Int(args[1])
    var rows = Int(args[2])
    var features = Int(args[3])
    var X = nm.ones[DType.float64](nm.Shape(rows, features))
    var y = List[Int]()
    for i in range(rows):
        y.append(i % 3)
    var model = MultinomialNB()
    _ = model.fit(X, y)
    var start = perf_counter_ns()
    for _ in range(loops):
        _ = model.fit(X, y)
    var elapsed = perf_counter_ns() - start
    print(Float64(elapsed) * 1e-9, model.feature_count_.item(0, 0))
