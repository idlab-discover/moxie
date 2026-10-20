import numojo as nm
from moxie.neighbors import KDTree, NearestNeighbors
from std.testing import assert_equal

def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(5,1))
    for i in range(5): X.unsafe_set(i,Float64(i))
    var tree = KDTree(X,1)
    assert_equal(tree.query(X).indices[4],4)
    var arrays = tree.get_arrays()
    assert_equal(len(tree.idx_array),len(arrays.idx_array))
    assert_equal(len(tree.node_data),len(arrays.node_data))
    assert_equal(tree.node_bounds.shape[0],2)
    assert_equal(tree.leaf_size,1)
    assert_equal(tree.sum_weight,5)
    var nn = NearestNeighbors(2,algorithm="kd_tree",leaf_size=1)
    nn.fit(X)
    assert_equal(nn._fit_method,"kd_tree")
    assert_equal(nn.kneighbors().indices[0],1)
    print("PASS: assembled KDTree consumer")
