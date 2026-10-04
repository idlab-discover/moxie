"""Robust covariance and outlier detection on data with two gross outliers."""
import numojo as nm
from numojo import Shape
from moxie.covariance import MinCovDet, EllipticEnvelope


def main() raises:
    var X = nm.zeros[DType.float64](Shape(40, 2))
    for i in range(40):
        var t = Float64(i) / 40.0
        X.unsafe_set(2 * i, Float64((i * 13) % 17) / 17.0 - 0.5 + t)
        X.unsafe_set(2 * i + 1, Float64((i * 7) % 19) / 19.0 - 0.5 - t)
    X.unsafe_set(0, 25.0)
    X.unsafe_set(3, -30.0)
    var mcd = MinCovDet(random_state=0)
    mcd.fit(X)
    print("robust location_:", mcd.location_().unsafe_get(0), mcd.location_().unsafe_get(1))
    var support = mcd.raw_support_()
    var count = 0
    for k in range(support.size):
        if support.unsafe_get(k):
            count += 1
    print("raw support size:", count)
    var detector = EllipticEnvelope(contamination=0.1, random_state=0)
    detector.fit(X)
    var labels = detector.predict(X)
    print("labels of samples 0, 1, 2:", labels.unsafe_get(0), labels.unsafe_get(1), labels.unsafe_get(2))
    print("offset_:", detector.offset_())
