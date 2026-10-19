from moxie.probe_a import checkA
from moxie.probe_b import checkB

def main() raises:
    checkA()
    checkB()
    print("Both independent estimator probes passed")
