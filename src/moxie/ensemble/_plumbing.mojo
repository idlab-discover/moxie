# Contract adaptation from scikit-learn; see LICENSES/scikit-learn.txt.
"""Small validation and ownership helpers, not sampling or aggregation engines."""
from std.sys import stderr
from numojo import NDArray
from moxie._random import RandomState
from moxie.ensemble._records import SamplingPlan, ClassAlignment


def snapshot[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    return X.copy()


def validate_sampling(plan: SamplingPlan) raises:
    if plan.n_samples < 1 or plan.n_features < 1:
        raise Error("ValueError: empty sampling population")
    if plan.max_samples < 1 or (not plan.bootstrap_samples and plan.max_samples > plan.n_samples):
        raise Error("ValueError: max_samples outside population")
    if plan.max_features < 1 or plan.max_features > plan.n_features:
        raise Error("ValueError: max_features outside population")


def additional_estimators(requested: Int, fitted: Int, warm_start: Bool) raises -> Int:
    if requested < 1 or fitted < 0:
        raise Error("ValueError: invalid estimator count")
    if not warm_start:
        return requested
    if requested < fitted:
        raise Error("ValueError: warm_start cannot remove estimators")
    return requested - fitted


def validate_oob(bootstrap: Bool, warm_start: Bool, bagging: Bool) raises:
    if not bootstrap:
        raise Error("ValueError: OOB requires bootstrap")
    if bagging and warm_start:
        raise Error("ValueError: bagging OOB requires warm_start=False")


def next_estimator_seed(mut rng: RandomState) raises -> Int:
    return rng.randint(0, 2147483647)


def validate_alignment(mapping: ClassAlignment, local_classes: Int) raises:
    if mapping.n_classes < 1 or mapping.columns.ndim != 1 or mapping.columns.shape[0] != local_classes:
        raise Error("ValueError: invalid class alignment shape")
    for i in range(local_classes):
        var column = Int(mapping.columns.item(i))
        if column < 0 or column >= mapping.n_classes:
            raise Error("ValueError: class column outside global classes")
        for j in range(i):
            if mapping.columns.item(j) == mapping.columns.item(i):
                raise Error("ValueError: duplicate class column")


def validate_names(names: List[String], active: List[Bool], reserved: List[String]) raises:
    if len(names) == 0 or len(names) != len(active):
        raise Error("ValueError: invalid named estimator list")
    var count = 0
    for i in range(len(names)):
        if names[i].__contains__("__"):
            raise Error("ValueError: estimator names cannot contain __")
        for j in range(i):
            if names[i] == names[j]:
                raise Error("ValueError: duplicate estimator name")
        for name in reserved:
            if names[i] == name:
                raise Error("ValueError: estimator name conflicts with parameter")
        if active[i]:
            count += 1
    if count == 0:
        raise Error("ValueError: all estimators are dropped")


def warn(mut warnings: List[String], message: String):
    """Emit a sklearn warning as `Category: message` on stderr and keep it."""
    print(message, file=stderr)
    warnings.append(message)
