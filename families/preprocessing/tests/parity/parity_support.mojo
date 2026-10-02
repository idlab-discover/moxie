"""Native side of the parity harness: binary inputs in, lossless text results out.

Floats print with shortest round-trip formatting, so Python parses them exactly.
"""
from numojo import NDArray, Shape, zeros
from std.sys import size_of


def load[d: DType](path: String, rows: Int, cols: Int) raises -> NDArray[d]:
    """Read a C-ordered binary array written by harness.py (rank 2, or rank 1 if cols < 0)."""
    var data: List[UInt8]
    with open(path, "r") as handle:
        data = handle.read_bytes()
    var size = rows * (cols if cols >= 0 else 1)
    comptime width = size_of[Scalar[d]]()
    if len(data) != size * width:
        raise Error("parity input " + path + " has unexpected size")
    var result = zeros[d](Shape(rows, cols)) if cols >= 0 else zeros[d](Shape(rows))
    var source = data.unsafe_ptr().bitcast[Scalar[d]]()
    for i in range(size):
        result.unsafe_set(i, source[i])
    return result^


def load_strings(path: String) raises -> List[String]:
    """One UTF-8 string per line; a lone NUL byte line denotes an empty string."""
    var text: String
    with open(path, "r") as handle:
        text = handle.read()
    var result = List[String]()
    if text.byte_length() == 0:
        return result^
    for line in text.split("\n"):
        result.append(String(line))
    _ = result.pop()
    return result^


def load_optional_strings(path: String) raises -> List[Optional[String]]:
    """Lines written by Driver.optional_strings: 'N' = missing, 'S<value>' = value."""
    var result = List[Optional[String]]()
    for line in load_strings(path):
        if line == "N":
            result.append(None)
        else:
            result.append(Optional[String](String(line.removeprefix("S"))))
    return result^


def emit[d: DType](name: String, x: NDArray[d]) raises:
    var rows = x.shape[0] if x.ndim >= 1 else 1
    var cols = x.shape[1] if x.ndim == 2 else -1
    var line = String("R ") + name + " " + String(x.ndim) + " " + String(rows) + " " + String(cols) + " " + String(d)
    var n = rows * (cols if cols >= 0 else 1)
    for k in range(n):
        line += " "
        var i = k // cols if cols > 0 else k
        var j = k % cols if cols > 0 else 0
        var offset = i * Int(x.strides[0]) + (j * Int(x.strides[1]) if x.ndim == 2 else 0)
        # Float32 text is not always round-trip exact in Mojo 1.0; the Float64
        # text of a Float32 value is, and parses back to the same Float32.
        comptime if d == DType.float32:
            line += String(Float64(x.unsafe_get(offset)))
        else:
            line += String(x.unsafe_get(offset))
    print(line)


def emit_scalar(name: String, value: Float64):
    print("F", name, String(value))


def emit_int(name: String, value: Int):
    print("I", name, value)


def emit_names(name: String, values: List[String]):
    print("S", name, len(values))
    for value in values:
        print(value)


def emit_error(name: String, error: Error):
    print("E", name, String(error).replace("\n", " "))
