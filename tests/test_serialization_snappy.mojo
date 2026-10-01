from std.testing import assert_equal, assert_true, assert_raises
from moxie.serialization import ModelArchive, dumps, loads, save, load, LoadLimits
from moxie.serialization_snappy import SnappyCodec


def main() raises:
    var archive = ModelArchive("tests.Large", 1)
    var shape: List[Int] = [10000]
    var values = List[Float64](length=10000, fill=42)
    archive.add_array[DType.float64]("large", shape, values)
    var plain = dumps(archive)
    var compressed = dumps[SnappyCodec](archive)
    assert_true(len(compressed) < len(plain))
    var restored = loads[SnappyCodec](compressed)
    assert_equal(dumps(restored), plain)
    var raw_restored = loads[SnappyCodec](plain)
    assert_equal(dumps(raw_restored), plain)
    save[SnappyCodec](archive, "builds/snappy.moxie")
    var disk = load[SnappyCodec]("builds/snappy.moxie")
    assert_equal(dumps(disk), plain)
    with assert_raises():
        _ = loads(compressed)
    with assert_raises():
        _ = loads[SnappyCodec](compressed, LoadLimits(decoded_bytes=79999))
    var bad: List[UInt8] = [128, 128, 128, 128, 15]
    with assert_raises():
        _ = SnappyCodec.decode(bad, 1)
    print("Snappy contracts passed; raw/compressed bytes:", len(plain), len(compressed))
