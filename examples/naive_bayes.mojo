"""Fit GaussianNB and incrementally train a count-based MultinomialNB."""
from moxie.base import clone
from moxie.naive_bayes import GaussianNB, MultinomialNB


def main() raises:
    var X: List[List[Float64]] = [
        [1.0, 2.0],
        [2.0, 1.0],
        [5.0, 6.0],
        [6.0, 5.0],
    ]
    var y: List[Int] = [0, 0, 1, 1]
    var gaussian = GaussianNB()
    _ = gaussian.fit(X, y)
    print("Gaussian predictions:", gaussian.predict(X))
    print("Gaussian accuracy:", gaussian.score(X, y))
    print("Clone is fitted:", clone(gaussian).__sklearn_is_fitted__())

    var batch1: List[List[Float64]] = [[3.0, 0.0], [0.0, 4.0]]
    var batch2: List[List[Float64]] = [[2.0, 1.0], [1.0, 3.0]]
    var labels: List[Int] = [0, 1]
    var classes: List[Int] = [0, 1]
    var counts = MultinomialNB()
    _ = counts.partial_fit(batch1, labels, classes=classes^)
    _ = counts.partial_fit(batch2, labels)
    print("Count predictions:", counts.predict(batch2))
    print("Count probabilities:", counts.predict_proba(batch2))
