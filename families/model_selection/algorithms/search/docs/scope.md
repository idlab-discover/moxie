# Dense search scope

Implemented: `ParameterGrid[V]` and `ParameterSampler[D]`. Grid values have one
copyable, deinitializable type `V`; a user-defined tagged record can represent
heterogeneous values. Grid input is `Dict[String, List[V]]` or a list of such
dictionaries. Names are sorted, the rightmost name cycles fastest, and grids
concatenate in input order. An empty dictionary yields one default candidate;
an empty grid list yields none. Empty value lists raise. Configurations and
returned candidates own independent copies. `len`, indexed access, `next` and
`reset` are supported. Negative indices explicitly raise.

`D` implements the family's existing `ParameterDistribution` and is copyable.
The associated `Value` now explicitly requires `Deinitable`, as storing sampled
values in owned configuration records requires destruction. Foundation contracts
are unchanged. `DiscreteDistribution[V]` supplies a simple typed distribution.

Finite sampling accepts a `ParameterGrid[D.Value]`. Mixed sampling accepts
`List[List[ParameterDomain[D]]]`; each domain has a name and either a value list
or a `D` distribution. All-list spaces sample without replacement using the
shared sklearn-compatible RNG helper, capped at grid size. When any distribution
is present, spaces are chosen uniformly with replacement, keys are visited in
sorted order, and distribution.sample(rng) supplies that parameter's value.
Different distribution types can be represented by a user-defined tagged `D`.

Supply `random_state=42` or a mutable `RandomState` as the third positional
argument. `None` requires the explicit stream overload. Candidate draws occur
eagerly at construction. `reset` replays the owned candidate snapshot without
new draws; construct another sampler for a new traversal of a mutable stream.
Python warnings about capped iteration counts are not emitted.

`GridSearchCV` and `RandomizedSearchCV` are implemented through the shared serial
CV engine. See [search interfaces and validation](search.md) for adapters, results,
refit modes, lifecycle, supported dense behavior and explicit native differences.
Python reflection and arbitrary heterogeneous dictionary dispatch are outside
this typed interface. Sparse estimator inputs/outputs remain deferred; candidate
enumeration has no sparse branch. No default requires sparse output.

Run `pixi run build`, `pixi run test` and `pixi run parity` here. The oracle checks
original ParameterGrid/ParameterSampler class bodies and compiles the unedited
pinned sklearn `_random.pyx` under ignored local builds. It compares exact
candidate values/order and subsequent RNG words (zero tolerance), including
sampling-ratio boundaries, mixed distributions and empty/default grids.
Reference: sklearn 4358542a1bd39af2e7e1cbace8b9fed067473c5b, BSD-3-Clause.

Search parity additionally runs original search/CV bodies against real estimators.
