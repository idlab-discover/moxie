import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.math import abs
from moxie.svm import l1_min_c, LinearSVC

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4,1))
    var y: List[Float64] = [0,0,1,1]
    for i in range(4):
        x.unsafe_set(i, Float64(i)-1.5)
    assert_equal(l1_min_c(x,y),0.125)
    assert_equal(l1_min_c(x,y,loss="log"),0.5)
    assert_equal(l1_min_c(x.astype[DType.float32](),y),0.125)
    var model = LinearSVC(penalty="l1",dual=False,C=0.124,random_state=0,tol=1e-10)
    _ = model.fit(x,y)
    assert_equal(model.coef_.unsafe_get(0),0.0)
    model.C = 0.126
    _ = model.fit(x,y)
    assert_true(abs(model.coef_.unsafe_get(0))>0)
    var zero = nm.zeros[DType.float64](nm.Shape(4,1))
    with assert_raises():
        _ = l1_min_c(zero,y)
    with assert_raises():
        _ = l1_min_c(x,y,loss="hinge")
    var one: List[String] = ["a","a","a","a"]
    assert_equal(l1_min_c(zero,one),0.125)
    print("PASS l1_min_c threshold and invalid inputs")
