import numojo as nm
from moxie.svm import LinearSVC, LinearSVR

def main() raises:
    var x = nm.zeros[DType.float64](nm.Shape(4, 1))
    var y: List[Float64] = [0, 0, 1, 1]
    var target = nm.zeros[DType.float64](nm.Shape(4))
    for i in range(4):
        x.unsafe_set(i, Float64(i) - 1.5)
        target.unsafe_set(i, Float64(i))
    var classifier = LinearSVC(random_state=42)
    _ = classifier.fit(x, y)
    print(classifier.predict(x))
    var strings = LinearSVC[String](random_state=42)
    var labels: List[String] = ["a", "a", "b", "b"]
    _ = strings.fit(x, labels)
    print(strings.predict(x))
    var regressor = LinearSVR(random_state=42)
    _ = regressor.fit(x, target)
    print(regressor.predict(x))
