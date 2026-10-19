from moxie.model_selection import ParameterGrid, ParameterSampler, DiscreteDistribution

def main() raises:
    var parameters = Dict[String, List[Int]]()
    parameters['depth'] = [2, 4, 8]
    parameters['seed'] = [0, 42]
    var grid = ParameterGrid[Int](parameters)
    var sampled = ParameterSampler[DiscreteDistribution[Int]](grid, 3, random_state=7)
    while True:
        var candidate = sampled.next()
        if not candidate:
            break
        print('depth:', candidate.value()['depth'], 'seed:', candidate.value()['seed'])
