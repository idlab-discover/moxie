# Composition contracts and validation

Moxie owns `moxie.pipeline`, `_pipeline`, `_pipeline_contracts`,
`_pipeline_sequential`, `_pipeline_union`, the factory scaffolds, and shared
`_feature_metadata` contracts. The implementations were integrated from
`moxie-pipeline` commit `5a948f1`. The scikit-learn reference remains
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`; BSD notices remain in the source
and `LICENSES/scikit-learn.txt`.

`NamedStep[E]` move-owns a named estimator. `CompositionStep` requires
constructor parameters to be copyable and destructible. `StepPair[A, B]`
provides a recursive heterogeneous product; clone delegates to each leaf's
clone hook. Configuration replacement preserves the supplied instance's state;
cloning normally produces unfitted children. `PairParams` contains both typed
parameter records. Updates execute in declaration order and are not transactional
on child errors. Unknown names fail before mutation.

Name/mode/row validation is shared in `_pipeline`. Names reject duplicates,
`__`, and constructor collisions. Empty names follow the reference and are
allowed. Pipeline has no drop mode; FeatureUnion supports drop and passthrough.
The execution engines retain child-owned learned state and use shared foundation
capability traits. Neither composition engine adds classifier semantics.

Feature metadata has two independent capabilities in `_feature_metadata`:

- `FeatureNamesOut.get_feature_names_out(input_features=None)` returns an owned
  list. `UnionFeatureNames` remains an alias, preserving existing constituents.
- `InputFeatureCount.get_n_features_in()` reads the constituent's learned width.
  The existing `UnionInputFeatures.n_features_in_()` method remains supported.

ColumnTransformer implements both shared contracts, retaining its public stored
`n_features_in_` attribute and a borrowing overload for list arguments. Omitting
names uses defaults; an explicit empty list is now validated against the input
width, matching sklearn. Compose's older `FeatureNamesOut` trait for component
transformers retains its list signature; column entries recognize either trait.
Nested ColumnTransformer names therefore propagate without an adapter.

Run `pixi run build`, `pixi run test`, and `pixi run parity` in Moxie. Pipeline
contracts and parity drivers live in `tests/pipeline`; Compose integration also
lives in `tests/compose/pipeline`. All use the same assembled namespace and
current Dummy implementation. The identical inherited foundation contracts run
once at the root. Negative compilation checks retain unsupported capabilities
and dtype/type mismatch coverage. Two isolated consumers retain shared dense
storage, independent copies, and ownership/export collision checks. Historical
source-hash drift rejection is superseded by live-source assembly; source receipts
still record the bytes consumed. No sibling Pipeline checkout is required.

The integrated real-estimator test combines ColumnTransformer branches,
FeatureUnion weighting, Pipeline, and LinearRegression in Float32/Float64.
Selection, feature counts, and names compare exactly against local sklearn;
predictions, refits, and R² use relative/absolute tolerances of `2e-5` for
Float32 and `1e-10` for Float64 to allow native least-squares rounding on this
small well-conditioned fixture. It also checks clone independence, unsupported
final feature names, and unfitted metadata. Raw comparisons and source hashes
remain under ignored `builds/tests/pipeline/compose-parity`.

Factory scaffolds, transformer-final/unsupervised-final pipelines, sparse
execution, fit-weight routing, persistence schemas, and metadata routing remain
unfinished. See [Pipeline](sequential.md) and [FeatureUnion](feature-union.md)
for their full implemented boundaries. Integration does not extend these claims.
