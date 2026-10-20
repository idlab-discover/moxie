"""Notebook data exchange for `%%mojo` cells (validation tooling only).

Arrays travel as `.npy` files: inputs shared from Python are read from
`MOXIE_NB_IN` and results are written to `MOXIE_NB_OUT`; the `%%mojo` magic
sets both. Reading and writing go through NumPy via Python interop, like
NuMojo's own `load`/`save`; estimators never see Python objects.
"""
from std.os import getenv
from std.python import Python, PythonObject

from numojo import NDArray
from numojo.core.layout import NDArrayShape
from numojo.routines.creation import array


def exchange_path(name: String, variable: String = "MOXIE_NB_IN") raises -> String:
    var base = getenv(variable)
    if not base:
        raise Error(variable + " is not set; run this program through %%mojo")
    return base + "/" + name + ".npy"


def _dtype_name[dtype: DType]() raises -> String:
    if dtype == DType.float64:
        return "float64"
    if dtype == DType.float32:
        return "float32"
    if dtype == DType.int64:
        return "int64"
    if dtype == DType.int32:
        return "int32"
    if dtype == DType.bool:
        return "bool"
    raise Error("Unsupported exchange dtype: " + String(dtype))


def load[dtype: DType = DType.float64](name: String) raises -> NDArray[dtype]:
    """Load a C-contiguous array shared from Python, cast to `dtype`."""
    var np = Python.import_module("numpy")
    var data = np.ascontiguousarray(
        np.load(exchange_path(name)), dtype=_dtype_name[dtype]()
    )
    var result = array[dtype](data=data)
    # NuMojo's array(data=...) drops every unit dimension ((500, 1) becomes
    # (500,)); restore NumPy's shape so one-column matrices stay rank 2.
    var shape = List[Int]()
    for i in range(Int(py=data.ndim)):
        shape.append(Int(py=data.shape[i]))
    if len(shape) != result.ndim:
        return result.reshape(NDArrayShape(shape))
    return result^


def load_strings(name: String) raises -> List[String]:
    """Load a 1-D array of string labels shared from Python."""
    var np = Python.import_module("numpy")
    var data = np.load(exchange_path(name))
    var result = List[String]()
    for i in range(Int(py=data.shape[0])):
        result.append(String(py=data[i].item()))
    return result^


def load_ints(name: String) raises -> List[Int]:
    """Load a 1-D integer array shared from Python as a List."""
    var values = load[DType.int64](name)
    var result = List[Int]()
    for i in range(values.size):
        result.append(Int(values.item(i)))
    return result^


def save[dtype: DType](name: String, values: NDArray[dtype]) raises:
    """Save an array for comparison in Python."""
    var np = Python.import_module("numpy")
    np.save(exchange_path(name, "MOXIE_NB_OUT"), values.to_numpy())


def save_floats(name: String, values: List[Float64]) raises:
    var np = Python.import_module("numpy")
    var out = Python.list()
    for value in values:
        out.append(value)
    np.save(exchange_path(name, "MOXIE_NB_OUT"), np.asarray(out, dtype="float64"))


def save_ints(name: String, values: List[Int]) raises:
    var np = Python.import_module("numpy")
    var out = Python.list()
    for value in values:
        out.append(value)
    np.save(exchange_path(name, "MOXIE_NB_OUT"), np.asarray(out, dtype="int64"))


def save_strings(name: String, values: List[String]) raises:
    var np = Python.import_module("numpy")
    var out = Python.list()
    for value in values:
        out.append(value)
    np.save(exchange_path(name, "MOXIE_NB_OUT"), np.asarray(out, dtype="str"))
