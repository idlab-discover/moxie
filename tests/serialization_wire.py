"""Independent wire-format checks using Python's struct/zlib (tooling only)."""
from pathlib import Path
import struct
import zlib

raw = Path('builds/roundtrip.moxie').read_bytes()
magic, version, size, checksum = struct.unpack_from('<8sQQQ', raw)
assert (magic, version, size) == (b'MOXIEMDL', 1, len(raw) - 32)
assert checksum == zlib.crc32(raw[32:])
pos = 32


def uint():
    global pos
    result, = struct.unpack_from('<Q', raw, pos)
    pos += 8
    return result


def blob():
    global pos
    n = uint()
    result = raw[pos:pos+n]
    assert len(result) == n
    pos += n
    return result


assert blob() == b'tests.Model'
assert uint() == 1
manifest = blob()
assert manifest[8:8 + struct.unpack_from('<Q', manifest)[0]].decode() == 'hello\0世界'
assert uint() == 1
assert blob() == b'weights'
assert (uint(), uint(), uint(), uint(), uint(), uint()) == (2, 2, 2, 2, 32, 0)
assert struct.unpack('<4Q', blob()) == (0, 1 << 63, 0x7ff8000000000042, 0x400c000000000000)
assert pos == len(raw)
print('independent wire layout and CRC checks passed')
