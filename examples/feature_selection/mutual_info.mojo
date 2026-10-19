"""Rank a nonlinear signal and select features using mutual information."""
import numojo as nm
from moxie._labels import StringTarget
from moxie.feature_selection import mutual_info_regression, MutualInfoScorer, SelectKBest


def main() raises:
    var X = nm.zeros[DType.float64](nm.Shape(40, 2))
    var y = nm.zeros[DType.float64](nm.Shape(40))
    for i in range(40):
        var signal = Float64(i - 20) / 10
        X.unsafe_set(i * 2, signal)
        X.unsafe_set(i * 2 + 1, Float64((i * 17) % 41))
        y.unsafe_set(i, signal * signal)
    print("MI scores in nats:", mutual_info_regression(X, y, random_state=42))
    var selector = SelectKBest(MutualInfoScorer(classification=False, random_state=42), k=1)
    _ = selector.fit(X, y.copy())
    print("Selected columns:", selector.get_support(True)[List[Int]])
    print("Selected matrix shape:", selector.transform(X).shape)

    # Native string classes paired with Float32 features.
    var messages = List[String]()
    for i in range(40):
        messages.append("spam" if i < 20 else "ham")
    var labels = StringTarget(messages)
    var X32 = X.astype[DType.float32]()
    var classifier = SelectKBest(
        MutualInfoScorer[DType.float32, target_type=StringTarget](random_state=42),
        k=1,
    )
    _ = classifier.fit(X32, labels.copy())
    print("Columns predictive of the string labels:", classifier.get_support(True)[List[Int]])
