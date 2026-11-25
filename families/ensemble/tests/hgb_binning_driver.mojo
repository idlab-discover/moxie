"""Read raw Float64 fixtures for hgb_binning_parity.py."""
from std.memory import bitcast
from std.sys import argv
from numojo import NDArray, Shape
from moxie.ensemble._hist_gradient_boosting.binning import _BinMapper


def read_array(fields: List[String], mut offset: Int, shape: Shape) raises -> NDArray[DType.float64]:
    var result = NDArray[DType.float64](shape)
    for i in range(result.size):
        result.unsafe_set(i, bitcast[DType.float64](Int64(atol(fields[offset]))))
        offset += 1
    return result^


def run_case(fields: List[String]) raises:
    var n_rows = atol(fields[0])
    var n_features = atol(fields[1])
    var n_query = atol(fields[2])
    var n_bins = atol(fields[3])
    var limit = atol(fields[4])
    var subsample: Optional[Int] = None
    if limit > 0:
        subsample = limit
    var weighted = fields[5] == "1"
    var offset = 6
    var data = read_array(fields, offset, Shape(n_rows, n_features))
    var query = read_array(fields, offset, Shape(n_query, n_features))
    var weights: Optional[NDArray[DType.float64]] = None
    if weighted:
        weights = read_array(fields, offset, Shape(n_rows))
    var domains = List[Optional[List[Float64]]](capacity=n_features)
    for _ in range(n_features):
        var count = atol(fields[offset])
        offset += 1
        if count < 0:
            domains.append(None)
        else:
            var categories = List[Float64](capacity=count)
            for _ in range(count):
                categories.append(Float64(atol(fields[offset])))
                offset += 1
            domains.append(categories^)

    var mapper = _BinMapper(n_bins=n_bins, subsample=subsample, random_state=42)
    var mapped = mapper.fit(data, known_categories=domains^, sample_weight=weights^).transform(query)
    var output = String(mapper.missing_values_bin_idx_)
    for feature in range(n_features):
        output += " | " + String(mapper.n_bins_non_missing_[feature])
        for threshold in mapper.bin_thresholds_[feature]:
            output += " " + String(bitcast[DType.int64](threshold))
    output += " |"
    for row in range(n_query):
        for feature in range(n_features):
            output += " " + String(mapped.unsafe_get(feature * n_query + row))
    var bitsets = mapper.make_known_categories_bitsets()
    output += " |"
    for index in bitsets[1]:
        output += " " + String(index)
    output += " |"
    for words in bitsets[0]:
        for word in words:
            output += " " + String(word)
    print(output)


def main() raises:
    for line in open(argv()[1], "r").read().split("\n"):
        if line.byte_length() == 0:
            continue
        var fields = List[String]()
        for field in line.split(" "):
            fields.append(String(field))
        run_case(fields)
