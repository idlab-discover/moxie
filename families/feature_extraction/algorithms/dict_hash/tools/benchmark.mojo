"""Native transform timings; pyperf orchestrates loops outside this process."""
from std.sys import argv
from std.time import perf_counter_ns
from moxie.feature_extraction import DictVectorizer
from moxie.feature_extraction._dict_vectorizer import FeatureBatch, FeatureRow, FeatureEntry, FeatureKey, FeatureValue


def bench[dtype: DType](rows: Int, columns: Int, loops: Int) raises:
    var batch = FeatureBatch()
    for i in range(rows):
        var row = FeatureRow()
        for j in range(columns):
            row.append(FeatureEntry(FeatureKey(String(j)), FeatureValue(Float64((i+j)%7))))
        batch.append(row^)
    var v = DictVectorizer[dtype](sparse=False)
    v.fit(batch)
    var checksum: Float64 = 0
    var start = perf_counter_ns()
    for k in range(loops):
        var result = v.transform(batch)
        checksum += Float64(result.unsafe_get(k % result.size))
    var elapsed = Float64(perf_counter_ns()-start)/1e9
    print(elapsed,checksum)


def main() raises:
    var args = argv()
    if args[1] == "float32":
        bench[DType.float32](Int(args[2]),Int(args[3]),Int(args[4]))
    else:
        bench[DType.float64](Int(args[2]),Int(args[3]),Int(args[4]))
