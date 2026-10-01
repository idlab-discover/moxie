"""Test-only fixture exchange for child parity drivers (never library code).

Line format written by tools/parity_io.py, one case per line:
    case|<id>|<bits>|<key>=<value>;...|<array>|<array>...
with arrays encoded as `<name>:<rows>:<cols>:<v1>,<v2>,...` (cols 0 = vector,
rows 0 = empty). Drivers print `result|<id>|<field>|...` or `error|<id>|<text>`.
"""
from numojo import NDArray, Shape, zeros
from std.sys import argv


@fieldwise_init
struct FixtureArray(Copyable, Movable):
    var rows: Int
    var cols: Int
    var values: List[Float64]

    def as_array[dtype: DType](self) raises -> NDArray[dtype]:
        if self.cols == 0:
            var a = zeros[dtype](Shape(self.rows))
            for i in range(self.rows):
                a.unsafe_set(i, self.values[i].cast[dtype]())
            return a^
        var a = zeros[dtype](Shape(self.rows, self.cols))
        for i in range(self.rows * self.cols):
            a.unsafe_set(i, self.values[i].cast[dtype]())
        return a^


struct Fixture(Movable):
    var id: String
    var bits: Int
    var params: Dict[String, String]
    var arrays: Dict[String, FixtureArray]

    def __init__(out self, line: String) raises:
        var fields = line.split("|")
        if len(fields) < 4 or String(fields[0]) != "case":
            raise Error("fixture: malformed line")
        self.id = String(fields[1])
        self.bits = Int(String(fields[2]))
        self.params = Dict[String, String]()
        self.arrays = Dict[String, FixtureArray]()
        var text = String(fields[3])
        if text.byte_length() > 0:
            for item in text.split(";"):
                var pair = item.split("=")
                self.params[String(pair[0])] = String(pair[1])
        for index in range(4, len(fields)):
            var parts = fields[index].split(":")
            var values = List[Float64]()
            var body = String(parts[3])
            if body.byte_length() > 0:
                for value in body.split(","):
                    values.append(atof(value))
            self.arrays[String(parts[0])] = FixtureArray(Int(String(parts[1])), Int(String(parts[2])), values^)

    def has(self, key: String) -> Bool:
        return key in self.params or key in self.arrays

    def text(self, key: String) raises -> String:
        return self.params[key]

    def number(self, key: String) raises -> Float64:
        return atof(self.params[key])

    def integer(self, key: String) raises -> Int:
        return Int(self.params[key])

    def flag(self, key: String) raises -> Bool:
        return self.params[key] == "1"

    def array[dtype: DType](self, key: String) raises -> NDArray[dtype]:
        return self.arrays[key].as_array[dtype]()


def encode[dtype: DType](name: String, a: NDArray[dtype]) raises -> String:
    """Encode a vector or matrix in the fixture array format (shortest repr)."""
    var rows = a.shape[0] if a.ndim >= 1 else 0
    var cols = a.shape[1] if a.ndim == 2 else 0
    var text = String(name, ":", rows, ":", cols, ":")
    var count = rows * (cols if a.ndim == 2 else 1)
    for i in range(count):
        if i > 0:
            text += ","
        var value: Float64
        if a.ndim == 2:
            value = Float64(a.unsafe_get((i // cols) * Int(a.strides[0]) + (i % cols) * Int(a.strides[1])))
        else:
            value = Float64(a.unsafe_get(i * Int(a.strides[0])))
        text += String(value)
    return text^


struct ResultLine(Movable):
    var text: String

    def __init__(out self, id: String):
        self.text = String("result|", id)

    def add[dtype: DType](mut self, name: String, a: NDArray[dtype]) raises:
        self.text += "|" + encode(name, a)

    def scalar(mut self, name: String, value: Float64):
        self.text += String("|", name, "=", value)

    def integer(mut self, name: String, value: Int):
        self.text += String("|", name, "=", value)

    def emit(self):
        print(self.text)


def report_error(id: String, error: Error):
    var message = String(error).replace("|", "/").replace("\n", " ")
    print(String("error|", id, "|", message))


def fixture_lines() raises -> List[String]:
    """Case lines from the file named by the first command-line argument."""
    var text = open(argv()[1], "r").read()
    var lines = List[String]()
    for line in text.splitlines():
        if line.startswith("case|"):
            lines.append(String(line))
    return lines^
