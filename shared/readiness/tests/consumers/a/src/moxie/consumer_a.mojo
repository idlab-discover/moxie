from moxie._random import RandomState
from moxie._readiness import workload, contracts
from std.sys import argv

def run() raises:
    var args = argv()
    contracts()
    var seed = Int(args[1])
    var state = RandomState(seed)
    for _ in range(Int(args[6])):
        workload(Int(args[2]), args[3], Int(args[4]) != 0, Int(args[5]) != 0, state)
