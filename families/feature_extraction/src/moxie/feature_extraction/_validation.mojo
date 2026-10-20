"""Dense capability and result-shape checks shared by family children."""
from numojo import NDArray


def require_dense_output(sparse: Bool = True) raises:
    if sparse:
        raise Error("NotImplementedError: sparse output deferred; select sparse=False")


def check_feature_matrix[dtype: DType](x: NDArray[dtype], rows: Int, columns: Int) raises:
    if rows < 0 or columns < 0:
        raise Error("ValueError: negative feature matrix extent")
    if x.ndim != 2:
        raise Error("ValueError: feature output must be rank two")
    if x.shape[0] != rows or x.shape[1] != columns:
        raise Error("ValueError: feature matrix shape disagrees with metadata")


def require_dense_graph_output(return_as: String = "sparse") raises:
    """Native return selector replacing Python array class objects."""
    if return_as == "dense":
        return
    if return_as == "sparse":
        raise Error("NotImplementedError: sparse graphs deferred; select return_as='dense'")
    raise Error("ValueError: return_as must be 'dense' or 'sparse'")
