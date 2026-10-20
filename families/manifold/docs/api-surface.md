# Public API scaffold

Implemented public exports `ClassicalMDS` and `trustworthiness` now use the
native implementations documented in [algorithms.md](algorithms.md). The
remaining declarations below are deferred.

Public names and callable parameter names/defaults follow the local scikit-learn
checkout. Newly declared constructors and operations raise `NotImplementedError`
through Mojo `Error`; they do not perform numerical work. Existing implementations
and family traits remain intact.

Mojo adaptations: heterogeneous inputs use `Some[Movable]`, accepting native
values without Python. Opaque configuration, learned attributes and deferred
results use the local `APIValue` marker. Python expression defaults that cannot
be expressed natively use `None`; the exact reference signature is recorded on
every declaration. Python `**kwargs` becomes an optional named native value
parameter. Implemented ClassicalMDS fitting and parameter updates return a mutable borrow
of the estimator, following the current Moxie contract. Deferred declarations
retain their original signatures and raise before performing any operation.
Properties are exposed as opaque fields. Inherited public methods are flattened
onto each struct; dynamic metadata-routing methods are outside this source-only
surface. Numerical behavior and Python reflection are not implemented.

The family build assembles current local modules and explicit public exports.
Framework and numerical sources come from their current local checkouts. `pixi run test`
checks that deferred estimator constructors raise, while `pixi run parity`
checks the implemented algorithms. See [algorithm support](algorithms.md).
