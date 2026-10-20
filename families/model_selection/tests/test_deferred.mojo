"""Deferred threshold and successive-halving APIs must fail clearly, never fit."""
from moxie.model_selection import (
    FixedThresholdClassifier,
    TunedThresholdClassifierCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)


def expect_not_implemented(message: String, name: String) raises:
    if not message.startswith("NotImplementedError: " + name + ".__init__"):
        raise Error("unclear deferred-API failure for " + name + ": " + message)


def main() raises:
    var grid = Dict[String, List[Float64]]()
    grid["C"] = [1.0, 10.0]
    try:
        _ = FixedThresholdClassifier(1, threshold=0.5)
        raise Error("FixedThresholdClassifier constructed")
    except e:
        expect_not_implemented(String(e), "FixedThresholdClassifier")
    try:
        _ = TunedThresholdClassifierCV(1)
        raise Error("TunedThresholdClassifierCV constructed")
    except e:
        expect_not_implemented(String(e), "TunedThresholdClassifierCV")
    try:
        _ = HalvingGridSearchCV(1, grid^)
        raise Error("HalvingGridSearchCV constructed")
    except e:
        expect_not_implemented(String(e), "HalvingGridSearchCV")
    try:
        _ = HalvingRandomSearchCV(1, 2)
        raise Error("HalvingRandomSearchCV constructed")
    except e:
        expect_not_implemented(String(e), "HalvingRandomSearchCV")
    print("PASS threshold and successive-halving APIs raise NotImplementedError")
