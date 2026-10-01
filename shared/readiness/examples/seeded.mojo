from moxie._random import RandomState
from moxie._readiness import workload

def main() raises:
    var state = RandomState(42)
    workload(4, "stratified", True, False, state)
