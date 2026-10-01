from std.testing import assert_equal, assert_true, assert_raises
from std.memory import bitcast
from moxie.serialization import BinaryWriter, BinaryReader, ModelArchive, LoadLimits, dumps, loads, save, load, crc32


def fixture() raises -> ModelArchive:
    var archive = ModelArchive("tests.Model", 1)
    var manifest = BinaryWriter()
    manifest.string("hello\x00世界")
    manifest.integer(-9223372036854775807)
    manifest.boolean(True)
    archive.manifest = manifest^.finish()
    var shape: List[Int] = [2, 2]
    var values: List[Float64] = [0.0, -0.0, bitcast[DType.float64, 1](UInt64(0x7ff8000000000042)), 3.5]
    archive.add_array[DType.float64]("weights", shape, values)
    return archive^


def put_uint(mut data: List[UInt8], offset: Int, value: UInt64):
    for i in range(8):
        data[offset + i] = UInt8((value >> UInt64(8 * i)) & 255)


def malformed_structure(bytes: List[UInt8]) raises:
    var reader = BinaryReader(Span(bytes))
    _ = reader.take(32)
    _ = reader.string()
    var schema = reader.position
    _ = reader.uint()
    var manifest_size = reader.position
    _ = reader.bytes(8388608)
    var count = reader.position
    _ = reader.uint()
    _ = reader.string()
    var dtype = reader.position
    _ = reader.uint()
    var rank = reader.position
    _ = reader.uint()
    var dim = reader.position
    _ = reader.uint()
    _ = reader.uint()
    var decoded_size = reader.position
    _ = reader.uint()
    var codec = reader.position
    _ = reader.uint()
    var encoded_size = reader.position
    var offsets: List[Int] = [schema, manifest_size, count, dtype, rank, dim, decoded_size, codec, encoded_size]
    var values: List[UInt64] = [0, 0xffffffffffffffff, 1025, 7, 33, 0xffffffffffffffff, 31, 42, 0xffffffffffffffff]
    for i in range(len(offsets)):
        var bad = bytes.copy()
        put_uint(bad, offsets[i], values[i])
        put_uint(bad, 24, UInt64(crc32(bad[32:])))
        with assert_raises():
            _ = loads(bad)
    var utf8 = bytes.copy()
    utf8[40] = 255
    put_uint(utf8, 24, UInt64(crc32(utf8[32:])))
    with assert_raises():
        _ = loads(utf8)


def array_types[dtype: DType]() raises:
    var archive = ModelArchive("tests.Types", 1)
    var shape: List[Int] = [2]
    var values: List[Scalar[dtype]] = [0, 17]
    archive.add_array[dtype]("typed", shape, values)
    var empty: List[Scalar[dtype]] = []
    var empty_shape: List[Int] = [0, 100]
    archive.add_array[dtype]("empty", empty_shape, empty)
    var bytes = dumps(archive)
    var restored = loads(bytes)
    var payload = restored.array("typed")
    assert_equal(payload.values[dtype](), values)
    var empty_payload = restored.array("empty")
    assert_equal(len(empty_payload.values[dtype]()), 0)


def main() raises:
    assert_equal(crc32(String("123456789").as_bytes()), UInt32(0xcbf43926))
    array_types[DType.float32]()
    array_types[DType.float64]()
    array_types[DType.int32]()
    array_types[DType.int64]()
    array_types[DType.uint32]()
    array_types[DType.uint64]()
    var archive = fixture()
    var bytes = dumps(archive)
    malformed_structure(bytes)
    var restored = loads(bytes)
    restored.expect("tests.Model", 1)
    var manifest = BinaryReader(Span(restored.manifest))
    assert_equal(manifest.string(), "hello\x00世界")
    assert_equal(manifest.integer(), -9223372036854775807)
    assert_true(manifest.boolean())
    manifest.finish()
    var payload = restored.array("weights")
    var values = payload.values[DType.float64]()
    assert_equal(bitcast[DType.uint64, 1](values[1]), UInt64(0x8000000000000000))
    assert_equal(bitcast[DType.uint64, 1](values[2]), UInt64(0x7ff8000000000042))
    assert_equal(values[3], 3.5)
    assert_equal(dumps(restored), bytes)
    save(archive, "builds/roundtrip.moxie")
    var disk = load("builds/roundtrip.moxie")
    assert_equal(dumps(disk), bytes)
    with assert_raises():
        _ = payload.values[DType.float32]()
    with assert_raises():
        restored.expect("wrong", 1)
    with assert_raises():
        restored.expect("tests.Model", 2)
    with assert_raises():
        _ = restored.array("missing")
    for end in range(len(bytes)):
        with assert_raises():
            _ = loads(bytes[:end])
    for i in range(len(bytes)):
        var bad = bytes.copy()
        bad[i] ^= 1
        with assert_raises():
            _ = loads(bad)
    var extra = bytes.copy()
    extra.append(0)
    with assert_raises():
        _ = loads(extra)
    with assert_raises():
        _ = loads(bytes, LoadLimits(decoded_bytes=31))
    with assert_raises():
        _ = loads(bytes, LoadLimits(manifest_bytes=1))
    with assert_raises():
        _ = loads(bytes, LoadLimits(payloads=0))
    with assert_raises():
        _ = loads(bytes, LoadLimits(rank=1))
    with assert_raises():
        _ = load("builds/roundtrip.moxie", LoadLimits(file_bytes=32))
    with assert_raises():
        _ = loads(bytes, LoadLimits(file_bytes=-1))
    archive.payloads.append(payload.copy())
    with assert_raises():
        _ = dumps(archive)
    print("serialization contracts passed")
