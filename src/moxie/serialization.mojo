"""Experimental, bounded native model container; family adapters own state schemas."""
from std.sys import size_of
from std.memory import bitcast


struct BinaryWriter(Movable):
    var data: List[UInt8]

    def __init__(out self):
        self.data = List[UInt8]()

    def finish(deinit self) -> List[UInt8]:
        return self.data^

    def uint(mut self, value: UInt64):
        for shift in range(0, 64, 8):
            self.data.append(UInt8((value >> UInt64(shift)) & 255))

    def integer(mut self, value: Int):
        self.uint(UInt64(bitcast[DType.uint64, 1](Int64(value))))

    def boolean(mut self, value: Bool):
        self.uint(UInt64(Int(value)))

    def raw(mut self, value: Span[UInt8, _]):
        self.data.extend(value)

    def bytes(mut self, value: Span[UInt8, _]):
        self.uint(UInt64(len(value)))
        self.raw(value)

    def string(mut self, value: String):
        self.bytes(value.as_bytes())

    def scalar[dtype: DType](mut self, value: Scalar[dtype]):
        comptime if size_of[Scalar[dtype]]() == 8:
            self.uint(bitcast[DType.uint64, 1](value))
        elif size_of[Scalar[dtype]]() == 4:
            var bits = bitcast[DType.uint32, 1](value)
            for shift in range(0, 32, 8):
                self.data.append(UInt8((bits >> UInt32(shift)) & 255))
        else:
            comptime assert False, "serialization supports 32/64-bit scalars"


struct BinaryReader[origin: Origin](Movable):
    var data: Span[UInt8, Self.origin]
    var position: Int

    def __init__(out self, data: Span[UInt8, Self.origin]):
        self.data = data
        self.position = 0

    def remaining(self) -> Int:
        return len(self.data) - self.position

    def take(mut self, count: Int) raises -> Span[UInt8, Self.origin]:
        if count < 0 or count > self.remaining():
            raise Error("ModelFormatError: truncated section")
        var start = self.position
        self.position += count
        return self.data[start:self.position]

    def uint(mut self) raises -> UInt64:
        var bytes = self.take(8)
        var value = UInt64(0)
        for i in range(8):
            value |= UInt64(bytes[i]) << UInt64(8 * i)
        return value

    def count(mut self, limit: Int) raises -> Int:
        var value = self.uint()
        if limit < 0 or value > UInt64(limit):
            raise Error("ModelFormatError: size limit exceeded")
        return Int(value)

    def integer(mut self) raises -> Int:
        return Int(bitcast[DType.int64, 1](self.uint()))

    def boolean(mut self) raises -> Bool:
        return self.count(1) == 1

    def bytes(mut self, limit: Int) raises -> Span[UInt8, Self.origin]:
        var count = self.count(limit)
        return self.take(count)

    def string(mut self, limit: Int = 1048576) raises -> String:
        return String(from_utf8=self.bytes(limit))

    def scalar[dtype: DType](mut self) raises -> Scalar[dtype]:
        comptime if size_of[Scalar[dtype]]() == 8:
            return bitcast[dtype, 1](self.uint())
        elif size_of[Scalar[dtype]]() == 4:
            var bytes = self.take(4)
            var bits = UInt32(0)
            for i in range(4):
                bits |= UInt32(bytes[i]) << UInt32(8 * i)
            return bitcast[dtype, 1](bits)
        else:
            comptime assert False, "serialization supports 32/64-bit scalars"

    def finish(self) raises:
        if self.remaining() != 0:
            raise Error("ModelFormatError: trailing data")


def crc32(data: Span[UInt8, _]) -> UInt32:
    """IEEE CRC-32 for accidental corruption detection, not authentication."""
    var crc = UInt32(0xFFFFFFFF)
    for byte in data:
        crc ^= UInt32(byte)
        for _ in range(8):
            crc = (crc >> 1) ^ (UInt32(0xEDB88320) if crc & 1 else UInt32(0))
    return crc ^ UInt32(0xFFFFFFFF)


trait PayloadCodec:
    @staticmethod
    def identifier() -> Int: ...

    @staticmethod
    def encode(data: Span[UInt8, _]) raises -> List[UInt8]: ...

    @staticmethod
    def decode(data: Span[UInt8, _], size: Int) raises -> List[UInt8]: ...


struct RawCodec(PayloadCodec):
    @staticmethod
    def identifier() -> Int:
        return 0

    @staticmethod
    def encode(data: Span[UInt8, _]) raises -> List[UInt8]:
        var result = List[UInt8]()
        result.extend(data)
        return result^

    @staticmethod
    def decode(data: Span[UInt8, _], size: Int) raises -> List[UInt8]:
        if len(data) != size:
            raise Error("ModelFormatError: raw payload size mismatch")
        return Self.encode(data)


struct LoadLimits(Copyable, Movable):
    var file_bytes: Int
    var decoded_bytes: Int
    var manifest_bytes: Int
    var payloads: Int
    var rank: Int

    def __init__(out self, *, file_bytes: Int = 268435456, decoded_bytes: Int = 268435456, manifest_bytes: Int = 8388608, payloads: Int = 1024, rank: Int = 32):
        self.file_bytes = file_bytes
        self.decoded_bytes = decoded_bytes
        self.manifest_bytes = manifest_bytes
        self.payloads = payloads
        self.rank = rank

    def validate(self) raises:
        if self.file_bytes > 4611686018427387904 or self.file_bytes < 32 or self.decoded_bytes < 0 or self.manifest_bytes < 0 or self.payloads < 0 or self.rank < 0:
            raise Error("ValueError: invalid model load limits")


def dtype_code[dtype: DType]() -> Int:
    comptime if dtype == DType.float32:
        return 1
    elif dtype == DType.float64:
        return 2
    elif dtype == DType.int32:
        return 3
    elif dtype == DType.int64:
        return 4
    elif dtype == DType.uint32:
        return 5
    elif dtype == DType.uint64:
        return 6
    else:
        comptime assert False, "unsupported serialized dtype"


def _element_bytes(code: Int) raises -> Int:
    if code < 1 or code > 6:
        raise Error("ModelFormatError: unknown dtype")
    return 4 if code == 1 or code == 3 or code == 5 else 8


def _array_bytes(code: Int, shape: Span[Int, _], limit: Int) raises -> Int:
    var count = _element_bytes(code)
    # Validate every dimension even when an earlier dimension is zero.
    var empty = False
    for dim in shape:
        if dim < 0 or dim > 2147483647:
            raise Error("ModelFormatError: invalid dimension")
        empty = empty or dim == 0
    if empty:
        return 0
    for dim in shape:
        if count > limit // dim:
            raise Error("ModelFormatError: array size limit exceeded")
        count *= dim
    if count > limit:
        raise Error("ModelFormatError: array size limit exceeded")
    return count


@fieldwise_init
struct ArrayPayload(Copyable, Movable):
    var name: String
    var dtype: Int
    var shape: List[Int]
    var data: List[UInt8]

    def values[dtype: DType](self) raises -> List[Scalar[dtype]]:
        if self.dtype != dtype_code[dtype]() or _array_bytes(self.dtype, self.shape, len(self.data)) != len(self.data):
            raise Error("ModelFormatError: array dtype or shape mismatch")
        var reader = BinaryReader(Span(self.data))
        var result = List[Scalar[dtype]]()
        while reader.remaining():
            result.append(reader.scalar[dtype]())
        return result^


struct ModelArchive(Copyable, Movable):
    var type_id: String
    var schema_version: Int
    var manifest: List[UInt8]
    var payloads: List[ArrayPayload]

    def __init__(out self, type_id: String, schema_version: Int):
        self.type_id = type_id
        self.schema_version = schema_version
        self.manifest = List[UInt8]()
        self.payloads = List[ArrayPayload]()

    def expect(self, type_id: String, schema_version: Int) raises:
        if self.type_id != type_id or self.schema_version != schema_version:
            raise Error("ModelFormatError: unsupported estimator type or state schema")

    def array(self, name: String) raises -> ArrayPayload:
        for payload in self.payloads:
            if payload.name == name:
                return payload.copy()
        raise Error("ModelFormatError: missing array " + name)

    def add_array[dtype: DType](mut self, name: String, shape: Span[Int, _], values: Span[Scalar[dtype], _]) raises:
        var writer = BinaryWriter()
        for value in values:
            writer.scalar[dtype](value)
        var dims = List[Int]()
        dims.extend(shape)
        self.payloads.append(ArrayPayload(name, dtype_code[dtype](), dims^, writer^.finish()))

    def validate(self, limits: LoadLimits = LoadLimits()) raises:
        limits.validate()
        if self.type_id.byte_length() == 0 or self.type_id.byte_length() > 4096 or self.schema_version < 1:
            raise Error("ModelFormatError: invalid type or schema")
        if len(self.manifest) > limits.manifest_bytes or len(self.payloads) > limits.payloads:
            raise Error("ModelFormatError: manifest or payload count limit exceeded")
        var total = 0
        var names = Dict[String, Bool]()
        for payload in self.payloads:
            if payload.name.byte_length() == 0 or payload.name.byte_length() > 4096 or payload.name in names:
                raise Error("ModelFormatError: invalid or duplicate payload name")
            names[payload.name] = True
            if len(payload.shape) > limits.rank:
                raise Error("ModelFormatError: rank limit exceeded")
            var size = _array_bytes(payload.dtype, payload.shape, limits.decoded_bytes - total)
            if size != len(payload.data):
                raise Error("ModelFormatError: payload shape/size mismatch")
            total += size


def dumps[Codec: PayloadCodec = RawCodec](archive: ModelArchive, limits: LoadLimits = LoadLimits()) raises -> List[UInt8]:
    archive.validate(limits)
    var body = BinaryWriter()
    body.string(archive.type_id)
    body.uint(UInt64(archive.schema_version))
    body.bytes(archive.manifest)
    body.uint(UInt64(len(archive.payloads)))
    for payload in archive.payloads:
        body.string(payload.name)
        body.uint(UInt64(payload.dtype))
        body.uint(UInt64(len(payload.shape)))
        for dim in payload.shape:
            body.uint(UInt64(dim))
        body.uint(UInt64(len(payload.data)))
        var packed = Codec.encode(payload.data)
        # Compression is opt-in, and a payload is compressed only if smaller.
        if Codec.identifier() != 0 and len(packed) < len(payload.data):
            body.uint(UInt64(Codec.identifier()))
            body.bytes(packed)
        else:
            body.uint(0)
            body.bytes(payload.data)
        if len(body.data) > limits.file_bytes - 32:
            raise Error("ModelFormatError: file size limit exceeded")
    if len(body.data) > limits.file_bytes - 32:
        raise Error("ModelFormatError: file size limit exceeded")
    var result = BinaryWriter()
    result.raw(String("MOXIEMDL").as_bytes())
    result.uint(1)
    result.uint(UInt64(len(body.data)))
    result.uint(UInt64(crc32(body.data)))
    result.raw(body.data)
    return result^.finish()


def loads[Codec: PayloadCodec = RawCodec](data: Span[UInt8, _], limits: LoadLimits = LoadLimits()) raises -> ModelArchive:
    limits.validate()
    if len(data) > limits.file_bytes:
        raise Error("ModelFormatError: file size limit exceeded")
    var reader = BinaryReader(data)
    if String(from_utf8=reader.take(8)) != "MOXIEMDL" or reader.uint() != 1:
        raise Error("ModelFormatError: unknown format or version")
    var size = reader.count(limits.file_bytes - 32)
    var checksum = reader.uint()
    var bytes = reader.take(size)
    reader.finish()
    if UInt64(crc32(bytes)) != checksum:
        raise Error("ModelFormatError: checksum mismatch")
    var body = BinaryReader(bytes)
    var type_id = body.string(4096)
    var version = body.count(2147483647)
    var result = ModelArchive(type_id, version)
    result.manifest.extend(body.bytes(limits.manifest_bytes))
    var count = body.count(limits.payloads)
    var total = 0
    for _ in range(count):
        var name = body.string(4096)
        var dtype = body.count(6)
        var rank = body.count(limits.rank)
        var shape = List[Int]()
        for _ in range(rank):
            shape.append(body.count(2147483647))
        var size = body.count(limits.decoded_bytes - total)
        if _array_bytes(dtype, shape, limits.decoded_bytes - total) != size:
            raise Error("ModelFormatError: array shape/size mismatch")
        total += size
        var codec = body.count(2147483647)
        var packed = body.bytes(limits.file_bytes)
        var unpacked: List[UInt8]
        if codec == 0:
            unpacked = RawCodec.decode(packed, size)
        elif codec == Codec.identifier():
            unpacked = Codec.decode(packed, size)
        else:
            raise Error("ModelFormatError: unsupported compression codec")
        if len(unpacked) != size:
            raise Error("ModelFormatError: decoded size mismatch")
        result.payloads.append(ArrayPayload(name, dtype, shape^, unpacked^))
    body.finish()
    result.validate(limits)
    return result^


def save[Codec: PayloadCodec = RawCodec](archive: ModelArchive, path: String, limits: LoadLimits = LoadLimits()) raises:
    """Validate/encode before opening; I/O failure can leave a partial file."""
    var data = dumps[Codec](archive, limits)
    with open(path, "w") as file:
        file.write_all(data)


def load[Codec: PayloadCodec = RawCodec](path: String, limits: LoadLimits = LoadLimits()) raises -> ModelArchive:
    limits.validate()
    with open(path, "r") as file:
        # Read in bounded chunks, including one byte past the limit to reject it.
        var data = List[UInt8]()
        while len(data) <= limits.file_bytes:
            var count = min(65536, limits.file_bytes - len(data) + 1)
            var chunk = file.read_bytes(count)
            if len(chunk) == 0:
                break
            data.extend(Span(chunk))
        return loads[Codec](data, limits)


trait ModelSchema:
    """Opt-in reconstruction adapter implemented and versioned by a family."""
    comptime Model: Movable

    @staticmethod
    def encode(model: Self.Model) raises -> ModelArchive: ...

    @staticmethod
    def decode(archive: ModelArchive) raises -> Self.Model: ...


def save_model[Schema: ModelSchema, Codec: PayloadCodec = RawCodec](model: Schema.Model, path: String, limits: LoadLimits = LoadLimits()) raises:
    var archive = Schema.encode(model)
    save[Codec](archive, path, limits)


def load_model[Schema: ModelSchema, Codec: PayloadCodec = RawCodec](path: String, limits: LoadLimits = LoadLimits()) raises -> Schema.Model:
    var archive = load[Codec](path, limits)
    return Schema.decode(archive)
