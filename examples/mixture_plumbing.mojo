"""Inspect a dense mixture layout without pretending to fit a model."""
from moxie.mixture._contracts import EMControl, covariance_shape

def main() raises:
    var config = EMControl()
    config.n_components = 3
    config.covariance_type = "diag"
    config.validate()
    print("Covariance layout:", covariance_shape(config.covariance_type, 3, 4))
    print("Fresh-fit restarts:", config.restart_count(False))
