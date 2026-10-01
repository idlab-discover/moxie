"""Optional raw Snappy payload codec, backed by the local mojo-snappy package."""
from mojo_snappy import compress, decompress, uncompressed_length
from .serialization import PayloadCodec


struct SnappyCodec(PayloadCodec):
    @staticmethod
    def identifier() -> Int:
        return 1

    @staticmethod
    def encode(data: Span[UInt8, _]) raises -> List[UInt8]:
        return compress(data)

    @staticmethod
    def decode(data: Span[UInt8, _], size: Int) raises -> List[UInt8]:
        if uncompressed_length(data) != size:
            raise Error("ModelFormatError: Snappy decoded length mismatch")
        return decompress(data, max_output_bytes=size)
