from moxie.nb_consumer_a import data, check_model as check_a
from moxie.nb_consumer_b import consume, check_model as check_b
from moxie._naive_bayes_dense import owned_snapshot
from std.testing import assert_equal
from moxie.naive_bayes import MultinomialNB, BernoulliNB

def main() raises:
    var X = data()
    var independent = owned_snapshot(X)
    var shared = X.view()
    assert_equal(consume(shared), 6)
    assert_equal(X.item(0, 0), 5)
    assert_equal(independent.item(0, 0), 1)
    var composed_counts = MultinomialNB()
    var composed_binary = BernoulliNB()
    assert_equal(composed_counts.alpha, composed_binary.alpha)
    check_a()
    check_b()
    print("PASS: composed producers and estimators share Moxie-owned family and numerics")
