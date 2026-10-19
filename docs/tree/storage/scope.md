# Dense tree storage scope

`moxie.tree._tree` implements the private `TreeStorage` engine over the pinned
family `TreeData` and `Node` records. Supported: owned node/value allocation,
growth and shrink, independent snapshots, leaf IDs (`apply`), dense values
(`predict`), node depths and weighted feature importances. Numeric, categorical
bitset and categorical hash splits and both missing directions are supported.
Tests cover single/multiple outputs, unequal class counts and padded values,
empty batches, threshold equality, infinities, NaN, zero root weight, malformed
topology, ownership, and contiguous/strided Float32 inputs. Training, pruning,
label encoding and estimator behavior belong to their separate owners.

Sparse inputs and CSR-returning `decision_path` are deferred, including
`decision_path` for dense input. No replacement dense path matrix is exposed.
Reference: sklearn/tree/_tree.pyx `_apply_sparse_csr`, `decision_path`, and
sklearn/tree/_classes.py `BaseDecisionTree.decision_path`, revision
4358542a1bd39af2e7e1cbace8b9fed067473c5b. The inherited legacy `TreeInspection`
trait does not imply sparse support; consumers use `DenseTreeInspection`.
Python serialization/reflection and bindings are separate work. The pinned
numerics operation exclusions continue to apply; no new generic operation or
container is introduced. There are no unsupported defaults requiring a dense
selection in this engine: inputs are typed dense arrays.

Validation: `pixi run tree-build`, `pixi run tree-test`, `pixi run tree-parity`. The native tests
include rejection of incomplete/malformed trees, invalid category codes and
wrong input ranks/features. A negative compilation test verifies `decision_path`
is unavailable. Parity compiles the unchanged relevant Tree methods from the
pinned local sklearn source into a test-only Cython extension, checks revision
and source cleanliness, and compares manually constructed trees. It does not
use trained estimators or claim training parity. IDs/depths compare exactly;
Float64 values/importances use 2e-14 relative/absolute tolerance for reduction
roundoff, with explicit NaN equivalence for zero root weight.

See [storage.md](storage.md) for native API adaptations and ownership.
