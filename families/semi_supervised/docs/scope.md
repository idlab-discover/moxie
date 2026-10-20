# Dense semi-supervised scope

Implemented estimators are `LabelPropagation`, `LabelSpreading`, and
`SelfTrainingClassifier`. See [the API guide](api-surface.md) for native types,
constructor defaults, learned state, adapters, and language differences.

Graph estimators share one RBF/callable affinity implementation and one owned
probability normalizer. They support Float32/Float64 dense features, integer or
tagged string labels, hard and soft clamping, unseen-data probabilities,
weighted accuracy, configuration cloning, repeated fits, and strided inputs.
Self-training supports strict threshold and k-best selection, finite/zero/None
iteration limits, final supervised refitting, owned transduction state, and
capability-specific delegation. GaussianNB is consumed from its owner in Moxie.

Deferred branches are sparse feature/target arrays, sparse callable affinities,
sparse graph outputs, and `kernel="knn"`, even when the input features are dense.
These modes are not silently densified. Metadata routing and Python bindings
remain excluded.

`pixi run semi-supervised-test` from the Moxie root builds all family modules and both isolated consumers, runs native
estimator and inherited contract tests, and checks dependency HEAD resolution and
module/export collisions. `pixi run semi-supervised-parity` executes the exact local sklearn
algorithm sources and generates native numerical assertions. It records seeds,
source hashes, compiler flags, support-library versions, and output under
`builds/parity`. These checks establish the tested dense scope, not universal
scikit-learn parity. See the parity tool for fixtures and tolerances.

All dependency sources follow their local HEAD and working tree through
`tools/dependencies.py`. Builds record the consumed revisions and hashes;
family modules are live inputs. Runtime-unneeded RNG and historical readiness-evidence
projects are omitted from the build closure. Only the consumed numerical module
is required from the numerical owner, not its mutable development documents.
Numax is an explicitly adopted runtime for GaussianNB's numerical primitives.
No dependency source or shared cache is modified by builds.
