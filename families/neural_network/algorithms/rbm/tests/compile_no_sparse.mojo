# Intentionally invalid: sparse storage has no supported typed API.
from moxie._sparse import CSRMatrix
from moxie.neural_network import BernoulliRBM


def main() raises:
    var model = BernoulliRBM(random_state=42)
    model.fit(CSRMatrix())
