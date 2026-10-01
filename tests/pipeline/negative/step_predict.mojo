from moxie._pipeline import NamedStep
from test_plumbing import Probe

def main() raises:
    var step = NamedStep("config_only", Probe(1))
    _ = step.predict(1)
