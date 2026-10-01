"""Owned composition plumbing; no Pipeline or FeatureUnion execution engine."""
# Name/mode semantics adapted from scikit-learn pipeline.py/metaestimators.py.
# Copyright the scikit-learn developers; BSD-3-Clause, see LICENSES/scikit-learn.txt.
from moxie.base import BaseEstimator, clone


def validate_names(names: List[String], reserved: List[String]) raises:
    for i in range(len(names)):
        if "__" in names[i]:
            raise Error("ValueError: step names must not contain __")
        for key in reserved:
            if names[i] == key:
                raise Error("ValueError: step name conflicts with constructor parameter")
        for j in range(i):
            if names[i] == names[j]:
                raise Error("ValueError: step names must be unique")


def resolve_step(names: List[String], name: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == name:
            return i
    raise Error("ValueError: unknown step name")


struct StepMode:
    comptime estimator = 0
    comptime passthrough = 1
    comptime drop = 2


def validate_mode(mode: Int, *, union: Bool = False) raises:
    if mode == StepMode.estimator or mode == StepMode.passthrough:
        return
    if union and mode == StepMode.drop:
        return
    raise Error("ValueError: unsupported step mode")


trait CompositionStep(BaseEstimator, Deinitable):
    """Stored native steps have destructible constructor parameter records."""
    comptime Params: Copyable & Deinitable


struct NamedStep[E: CompositionStep](CompositionStep):
    """Move-owned estimator leaf. Name is container configuration, not E.Params."""
    comptime Params = Self.E.Params
    var name: String
    var estimator: Self.E

    def __init__(out self, name: String, var estimator: Self.E) raises:
        if "__" in name:
            raise Error("ValueError: step names must not contain __")
        self.name = name
        self.estimator = estimator^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        return self.estimator.get_params(deep)

    def set_params(mut self, params: Self.Params) raises:
        self.estimator.set_params(params)

    def set_named_params(mut self, name: String, params: Self.Params) raises:
        if name != self.name:
            raise Error("ValueError: unknown step name")
        self.set_params(params)

    def replace(mut self, var estimator: Self.E):
        self.estimator = estimator^

    def __sklearn_clone__(self) raises -> Self:
        return Self(self.name, clone(self.estimator))

    def __sklearn_is_fitted__(self) -> Bool:
        return self.estimator.__sklearn_is_fitted__()


@fieldwise_init
struct PairParams[A: Copyable & Deinitable, B: Copyable & Deinitable](Copyable, Movable):
    var first: Self.A
    var second: Self.B


struct StepPair[A: CompositionStep, B: CompositionStep](CompositionStep):
    """Recursive heterogeneous product; execution order belongs to its consumer."""
    comptime Params = PairParams[Self.A.Params, Self.B.Params]
    var first: Self.A
    var second: Self.B

    def __init__(out self, var first: Self.A, var second: Self.B):
        self.first = first^
        self.second = second^

    def get_params(self, deep: Bool = True) raises -> Self.Params:
        var first: Self.A.Params = self.first.get_params(deep)
        var second: Self.B.Params = self.second.get_params(deep)
        return Self.Params(first^, second^)

    def set_params(mut self, params: Self.Params) raises:
        self.first.set_params(params.first)
        self.second.set_params(params.second)

    def __sklearn_clone__(self) raises -> Self:
        var first: Self.A = clone(self.first)
        var second: Self.B = clone(self.second)
        return Self(first^, second^)

    def __sklearn_is_fitted__(self) -> Bool:
        return self.first.__sklearn_is_fitted__() and self.second.__sklearn_is_fitted__()


def require_nonempty_pipeline(count: Int) raises:
    if count <= 0:
        raise Error("ValueError: the pipeline is empty")


def check_union_rows(expected: Int, actual: Int) raises:
    if expected < 0 or actual != expected:
        raise Error("ValueError: union branches must preserve sample count")
