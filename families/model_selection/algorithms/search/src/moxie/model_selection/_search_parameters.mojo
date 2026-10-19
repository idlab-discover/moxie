"""Typed candidate enumeration; algorithm owner: model_selection.search."""
from moxie.model_selection._candidates import ParameterCandidates, ParameterDistribution
from moxie._random import RandomState


def _keys[V: Copyable & Deinitable](mapping: Dict[String, V]) -> List[String]:
    var names = List[String]()
    for key in mapping.keys():
        names.append(key.copy())
    for i in range(1, len(names)):
        var key = names[i].copy()
        var j = i
        while j > 0:
            if names[j - 1] <= key:
                break
            names[j] = names[j - 1].copy()
            j -= 1
        names[j] = key^
    return names^


struct ParameterGrid[V: Copyable & Deinitable](ParameterCandidates, Copyable, Sized):
    comptime Params = Dict[String, Self.V]
    var param_grid: List[Dict[String, List[Self.V]]]
    var position: Int
    var size: Int

    def __init__(out self, param_grid: Dict[String, List[Self.V]]) raises:
        var grids = List[Dict[String, List[Self.V]]]()
        grids.append(param_grid.copy())
        self = Self(grids)

    def __init__(out self, param_grid: List[Dict[String, List[Self.V]]]) raises:
        self.param_grid = param_grid.copy()
        self.position = 0
        self.size = 0
        for i in range(len(param_grid)):
            var count = 1
            for key in param_grid[i].keys():
                var n = len(param_grid[i][key])
                if n == 0:
                    raise Error("ValueError: parameter values must be nonempty")
                if count > 9223372036854775807 // n:
                    raise Error("OverflowError: parameter grid is too large")
                count *= n
            if self.size > 9223372036854775807 - count:
                raise Error("OverflowError: parameter grid is too large")
            self.size += count

    def __len__(self) -> Int:
        return self.size

    def __getitem__(self, index: Int) raises -> Self.Params:
        if index < 0 or index >= self.size:
            raise Error("IndexError: parameter grid index out of range")
        var offset = index
        for g in range(len(self.param_grid)):
            var names = _keys(self.param_grid[g])
            var count = 1
            for i in range(len(names)):
                count *= len(self.param_grid[g][names[i]])
            if offset >= count:
                offset -= count
                continue
            var params = Self.Params()
            for k in range(len(names)):
                var i = len(names) - 1 - k
                var n = len(self.param_grid[g][names[i]])
                params[names[i]] = self.param_grid[g][names[i]][offset % n].copy()
                offset //= n
            return params^
        raise Error("IndexError: parameter grid index out of range")

    def next(mut self) raises -> Optional[Self.Params]:
        if self.position == self.size:
            return None
        var result = self[self.position]
        self.position += 1
        return result^

    def reset(mut self):
        self.position = 0


struct DiscreteDistribution[V: Copyable & Deinitable](ParameterDistribution, Copyable):
    """Typed example distribution; finite grid axes use lists instead."""
    comptime Value = Self.V
    var values: List[Self.V]

    def __init__(out self, values: List[Self.V]) raises:
        if len(values) == 0:
            raise Error("ValueError: distribution values must be nonempty")
        self.values = values.copy()

    def sample(self, mut random_state: RandomState) raises -> Self.Value:
        return self.values[random_state.randint(len(self.values))].copy()


struct ParameterDomain[D: ParameterDistribution & Copyable & Deinitable](Copyable, Movable):
    var name: String
    var values: List[Self.D.Value]
    var distribution: Optional[Self.D]

    def __init__(out self, name: String, values: List[Self.D.Value]) raises:
        if len(values) == 0:
            raise Error("ValueError: parameter values must be nonempty")
        self.name = name
        self.values = values.copy()
        self.distribution = None

    def __init__(out self, name: String, distribution: Self.D):
        self.name = name
        self.values = List[Self.D.Value]()
        self.distribution = distribution.copy()


struct ParameterSampler[D: ParameterDistribution & Copyable & Deinitable](ParameterCandidates, Sized):
    comptime Params = Dict[String, Self.D.Value]
    var candidates: List[Self.Params]
    var position: Int
    var n_iter: Int
    var random_state: Optional[Int]

    def __init__(out self, param_distributions: ParameterGrid[Self.D.Value], n_iter: Int, *, random_state: Optional[Int] = None) raises:
        if not random_state:
            raise Error("ValueError: random_state=None requires an application-owned RandomState")
        var rng = RandomState(random_state.value())
        self = Self(param_distributions, n_iter, rng)
        self.random_state = random_state

    def __init__(out self, param_distributions: ParameterGrid[Self.D.Value], n_iter: Int, mut rng: RandomState) raises:
        if n_iter <= 0:
            raise Error("ValueError: n_iter must be positive")
        self.n_iter = n_iter
        self.random_state = None
        self.position = 0
        self.candidates = List[Self.Params]()
        var indices = rng.sample_without_replacement(len(param_distributions), min(n_iter, len(param_distributions)))
        from numojo import Item
        for i in range(indices.size):
            self.candidates.append(param_distributions[Int(indices[Item(i)])])

    def __init__(out self, param_distributions: List[List[ParameterDomain[Self.D]]], n_iter: Int, *, random_state: Optional[Int] = None) raises:
        if not random_state:
            raise Error("ValueError: random_state=None requires an application-owned RandomState")
        var rng = RandomState(random_state.value())
        self = Self(param_distributions, n_iter, rng)
        self.random_state = random_state

    def __init__(out self, param_distributions: List[List[ParameterDomain[Self.D]]], n_iter: Int, mut rng: RandomState) raises:
        if n_iter <= 0:
            raise Error("ValueError: n_iter must be positive")
        var spaces = param_distributions.copy()
        var all_lists = True
        var grids = List[Dict[String, List[Self.D.Value]]]()
        for g in range(len(spaces)):
            var grid = Dict[String, List[Self.D.Value]]()
            for i in range(len(spaces[g])):
                for j in range(i):
                    if spaces[g][i].name == spaces[g][j].name:
                        raise Error("ValueError: duplicate parameter name")
                all_lists = all_lists and not Bool(spaces[g][i].distribution)
                if not spaces[g][i].distribution:
                    grid[spaces[g][i].name] = spaces[g][i].values.copy()
            grids.append(grid^)
            for i in range(1, len(spaces[g])):
                var axis = spaces[g][i].copy()
                var j = i
                while j > 0:
                    if spaces[g][j - 1].name <= axis.name:
                        break
                    spaces[g][j] = spaces[g][j - 1].copy()
                    j -= 1
                spaces[g][j] = axis^
        if all_lists:
            self = Self(ParameterGrid[Self.D.Value](grids), n_iter, rng)
            return
        self.n_iter = n_iter
        self.random_state = None
        self.position = 0
        self.candidates = List[Self.Params]()
        for _ in range(n_iter):
            var g = rng.randint(len(spaces))
            var params = Self.Params()
            for i in range(len(spaces[g])):
                if spaces[g][i].distribution:
                    params[spaces[g][i].name] = spaces[g][i].distribution.value().sample(rng)
                else:
                    var j = rng.randint(len(spaces[g][i].values))
                    params[spaces[g][i].name] = spaces[g][i].values[j].copy()
            self.candidates.append(params^)

    def __len__(self) -> Int:
        return len(self.candidates)

    def next(mut self) raises -> Optional[Self.Params]:
        if self.position == len(self.candidates):
            return None
        var result = self.candidates[self.position].copy()
        self.position += 1
        return result^

    def reset(mut self):
        """Replay the owned candidate snapshot without consuming new draws."""
        self.position = 0
