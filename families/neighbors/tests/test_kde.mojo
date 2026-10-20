import numojo as nm
from std.testing import assert_equal, assert_true, assert_raises
from std.sys import argv
from moxie.base import clone
from moxie.neighbors import KernelDensity

def exercise[dtype: DType](kernel: String, weighted: Bool, rule: String, algorithm: String = "auto", metric: String = "euclidean") raises:
    var x = nm.zeros[dtype](nm.Shape(4,2))
    for i in range(4):
        x.unsafe_set(i*2,Float64(i).cast[dtype]())
        x.unsafe_set(i*2+1,Float64(i%2).cast[dtype]())
    var q = nm.zeros[dtype](nm.Shape(3,2))
    q.unsafe_set(2,1.5)
    q.unsafe_set(3,0.75)
    q.unsafe_set(4,20)
    var weights = nm.zeros[dtype](nm.Shape(4))
    weights.unsafe_set(0,1)
    weights.unsafe_set(2,3)
    weights.unsafe_set(3,2)
    var sw = Optional[nm.NDArray[dtype]](None)
    if weighted: sw = weights.view()
    var model = KernelDensity[dtype](bandwidth=1.5,kernel=kernel,bandwidth_rule=rule,algorithm=algorithm,metric=metric)
    with assert_raises(): _ = model.score_samples(q)
    model.fit(x,sample_weight=sw)
    var scores = model.score_samples(q)
    for i in range(3): print("S",scores.unsafe_get(i))
    print("T",model.score(q))
    print("B",model.bandwidth_)
    if kernel == "gaussian":
        var samples = model.sample(3,42)
        for i in range(6): print("G",samples.unsafe_get(i))
        var repeated = model.sample(3,42)
        for i in range(6): assert_equal(samples.unsafe_get(i),repeated.unsafe_get(i))
    var fresh = clone(model)
    assert_true(not fresh.__sklearn_is_fitted__())
    var invalid = nm.zeros[dtype](nm.Shape(0,2))
    with assert_raises(): model.fit(invalid)
    assert_true(model.__sklearn_is_fitted__())
    var zero = nm.zeros[dtype](nm.Shape(4))
    with assert_raises(): model.fit(x,sample_weight=zero.view())
    var bad = KernelDensity[dtype](algorithm="kd_tree")
    bad.fit(x)
    assert_equal(bad._fit_method,"kd_tree")
    var narrow = KernelDensity[dtype](bandwidth=0.01)
    narrow.fit(x)
    var far = narrow.score_samples(q)
    assert_true(far.unsafe_get(2) < -1000)
    var changed = model.get_params()
    changed.bandwidth = 2
    model.set_params(changed)
    model.fit(x)
    assert_true(model.bandwidth_ > 0)

def main() raises:
    var args = argv()
    var kernel = String("gaussian")
    var weighted = False
    var rule = String("")
    if len(args)>2: kernel=args[2]
    if len(args)>3: weighted=args[3]=="weighted"
    if len(args)>4: rule=args[4]
    var algorithm = String("auto")
    var metric = String("euclidean")
    if len(args)>5: algorithm=args[5]
    if len(args)>6: metric=args[6]
    if len(args)>1 and args[1]=="f32": exercise[DType.float32](kernel,weighted,rule,algorithm,metric)
    else: exercise[DType.float64](kernel,weighted,rule,algorithm,metric)
