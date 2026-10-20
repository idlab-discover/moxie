"""Native side of sampling_parity.py: one command per input line, one result line per command."""
import numojo as nm
from std.memory import bitcast
from std.sys import argv
from moxie._random import RandomState
from moxie.ensemble._records import Values, Indices
from moxie.ensemble._sampling import (
    MaxSamples, numpy_sum, get_n_samples_bootstrap, generate_sample_indices, generate_unsampled_indices,
    sample_counts, generate_bagging_indices, balanced_subsample_weight, balanced_class_weight,
)

def f64(token: String) raises -> Float64:
    return bitcast[DType.float64](Int64(atol(token)))

def bits(value: Float64) -> String:
    return String(bitcast[DType.int64](value))

def floats(fields: List[String], first: Int, count: Int) raises -> Values:
    var result = nm.zeros[DType.float64](nm.Shape(count))
    for i in range(count):
        result.unsafe_set(i, f64(fields[first + i]))
    return result^

def weights(fields: List[String], at: Int, n: Int) raises -> Optional[Values]:
    if fields[at] == "none":
        return None
    return floats(fields, at + 1, n)

def ints(values: Indices) raises -> String:
    var line = String()
    for i in range(values.size):
        line += " " + String(values.unsafe_get(i))
    return line

def float_bits(values: Values) raises -> String:
    var line = String()
    for i in range(values.size):
        line += " " + bits(values.unsafe_get(i))
    return line

def main() raises:
    var text = open(argv()[1], "r").read()
    for raw in text.split("\n"):
        if raw.byte_length() == 0:
            continue
        var fields = List[String]()
        for token in raw.split(" "):
            fields.append(String(token))
        ref tag = fields[0]
        if tag == "SUM":
            var n = atol(fields[1])
            print("SUM", bits(numpy_sum(floats(fields, 2, n))))
        elif tag == "NBOOT":  # n kind value weights
            var n = atol(fields[1])
            var option = MaxSamples(NoneType())
            if fields[2] == "int":
                option = MaxSamples(atol(fields[3]))
            elif fields[2] == "float":
                option = MaxSamples(f64(fields[3]))
            var warnings = List[String]()
            try:
                var count = get_n_samples_bootstrap(n, option, weights(fields, 4, n), warnings)
                print("NBOOT", count, len(warnings))
            except e:
                print("NBOOT error")
        elif tag == "BOOT":  # seed n n_bootstrap weights
            var seed = atol(fields[1])
            var n = atol(fields[2])
            var draws = atol(fields[3])
            var w = weights(fields, 4, n)
            try:
                var rows = generate_sample_indices(seed, n, draws, w)
                var oob = generate_unsampled_indices(seed, n, draws, w)
                print("BOOT" + ints(rows) + " |" + ints(oob) + " |" + ints(sample_counts(rows, n)))
            except e:
                print("BOOT error")
        elif tag == "BAG":  # seed bf bs n_features n_samples max_features max_samples weights
            var rng = RandomState(atol(fields[1]))
            var n = atol(fields[5])
            try:
                var drawn = generate_bagging_indices(rng, fields[2] == "1", fields[3] == "1", atol(fields[4]), n,
                    atol(fields[6]), atol(fields[7]), weights(fields, 8, n))
                print("BAG" + ints(drawn[0]) + " |" + ints(drawn[1]))
            except e:
                print("BAG error")
        elif tag == "SUB":  # n outputs m indices... y...
            var n = atol(fields[1])
            var outputs = atol(fields[2])
            var m = atol(fields[3])
            var indices = nm.zeros[DType.int64](nm.Shape(m))
            for i in range(m):
                indices.unsafe_set(i, Int64(atol(fields[4 + i])))
            var y = nm.zeros[DType.float64](nm.Shape(n, outputs))
            for i in range(n * outputs):
                y.unsafe_set(i, f64(fields[4 + m + i]))
            print("SUB" + float_bits(balanced_subsample_weight(y, indices)))
        elif tag == "BAL":  # n outputs y... weights
            var n = atol(fields[1])
            var outputs = atol(fields[2])
            var y = nm.zeros[DType.float64](nm.Shape(n, outputs))
            for i in range(n * outputs):
                y.unsafe_set(i, f64(fields[3 + i]))
            print("BAL" + float_bits(balanced_class_weight(y, weights(fields, 3 + n * outputs, n))))
