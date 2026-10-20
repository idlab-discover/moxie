"""Fit count models with dense arrays through the public module."""
import numojo as nm
from moxie.naive_bayes import MultinomialNB, ComplementNB

def main() raises:
    var X = nm.ones[DType.float64](nm.Shape(2, 2))
    X[nm.Item(0, 1)] = 0
    var y: List[Int] = [0, 1]
    var model = MultinomialNB()
    _ = model.fit(X, y)
    print(model.predict(X))
    var complement = ComplementNB(norm=True)
    _ = complement.fit(X, y)
    print(complement.predict_proba(X))
