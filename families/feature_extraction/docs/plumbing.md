# Feature extraction ownership and plumbing

Moxie owns this family, its shared contracts and both algorithm directories.
`algorithms/dict_hash` supplies `_dict_vectorizer`; `algorithms/image` supplies
`image`. Family `src` supplies contracts, text declarations and FeatureHasher's
raising declaration. The root Moxie manifest chooses these owners explicitly.
No standalone source path is consumed. Internal manifests keep isolated
contract consumers reproducible. Global framework, random and dense helpers
come from current Moxie sources; compatible toolchain locks remain unchanged.

The remaining sections describe extension contracts. Planned text/hash/TF-IDF
engines remain deferred; dictionary and image implementations now exist with
the supported limits in their child docs. Historical source-hash requirements
for Moxie sources are superseded by [family development](../../../docs/family-development.md).

## Storage and native compatibility

Import `NDArray`, `Shape`, `Item` directly from `numojo`; no family
numeric wrapper or local sparse substitute exists. Dtype is a compile-time
parameter. The current numerical policy approves Float32/Float64 storage and
operations listed in the shared [numerics policy](../../../shared/numerics/docs/numerics.md).
Future integer-valued dictionary output, image inputs, bool masks and integer
graphs use the same NDArray, with dtype-specific indexing/conversion/overflow
parity in their owning session. That is required dense algorithm work, not an
implicit exclusion. Metadata lists/maps below are not numeric array backends.

Borrow numerical inputs. Return owned arrays by move. `.copy()` makes an
independent learned snapshot; `.view()` explicitly aliases and retains buffer
lifetime. Strides count elements; offset/strided access belongs to NuMojo.
Slicing/reshape/transpose/casts may allocate; a strided copy may copy the entire
backing buffer. Do not introduce another shape/stride allocator. Fresh dense
outputs are C-order; accepting strided inputs must preserve logical order.
No concurrent mutation or overlapping in-place computation is promised.

Use `moxie._dense.sum_axis2d` and `row_squared_norms` for their approved matrix
operations. Global sum, explicit-output-dtype mean, finite elementwise multiply,
column subtraction and rank-2 products retain the numerical owner's limits.
NuMojo axis sum, dot, general solve/inverse/LU, variance/ddof, general batch or
vector matmul, and estimator RNG are excluded as general operations. New
operation evidence belongs at its owner. No numerical operation is replaced
or benchmark claim made here.

Mojo 1.0.0, MAX 26.5.0 and NuMojo 0.10.0 revision
`515fb2856f0ecf3d2740a34d958fe168183e1129` are pinned by the resolved environment
and the preserved external dependency metadata. Keep shared random builds at `-O3 --fp-mode=contract=off`.
Integer seeds reset at reference `check_random_state` call boundaries; mutable
`RandomState` is borrowed and advances. The native None equivalent is an
application-owned entropy context, as documented by the random owner. Sampling
patch positions draws all row positions then all columns, with replacement;
children must verify exact draw consumption, including singleton bounds.

## Mapping and column interfaces

`_mapping.category_key(name: String, value: String, separator: String = "=")
-> String` concatenates without escaping or Unicode normalization. Thus a raw
name `a=b` and categorical pair `a`, `b` intentionally identify the same column.
Empty separators/values are valid; strings are owned UTF-8. Dictionary key
identity must be preserved until the categorical naming step.

`OrderedVocabulary[Key: Copyable & Deinitable & Hashable & Equatable](keys: List[Key]) raises`
is an immutable snapshot of caller-selected column order. It copies the keys,
rejects duplicates, and stores `List[Key]` plus `Dict[Key, Int]`. `size() -> Int`,
`keys() -> List[Key]` (independent copy), and `column(key: Key) raises -> Int`
are the interface; -1 means unknown, never an insertion. Empty vocabularies are
valid. Copies are independent. Construction costs O(K) expected time/storage,
plus key bytes; lookups are expected O(1). The stdlib dictionary hash only
indexes metadata: its bucket layout never defines column order or feature hashes.

The algorithm supplies already ordered keys: `sort=True` uses reference key
ordering; `sort=False` uses first encounter order across rows and entries.
Unicode lexicographic comparison and heterogeneous keys need explicit native
adapters and parity. Generic typed keys allow native integer or string keys;
arbitrary Python objects/reflection remain binding work. `FeatureNames` is an
optional trait with `get_feature_names_out(self) raises -> List[String]`.
Typed string lists replace NumPy object-string arrays. The concrete algorithm
adds sklearn's ignored optional `input_features` overload; hashing and images
need not implement this capability. Display-string conversion must not conflate
numeric and string key identities in the vocabulary.

Raw mapping input is an ordered batch of typed mappings/entry iterables, with
key/value types supplied by the concrete estimator (the foundation's associated
Input type). Values include numeric scalars, categorical strings, iterables of
strings and the reference None conversion. Reject nested mappings and non-string
categorical iterable entries. The child defines a typed sum/overloads preserving
these cases rather than coercing every value to a string. The dictionary engine owns input decoding and vocabulary learning.

`_validation.require_dense_output(sparse: Bool = True) raises` rejects the
reference sparse default with an explicit error; False succeeds. It must run
before allocating output. `require_dense_graph_output(return_as: String = "sparse")
raises` accepts only `"dense"`, rejects `"sparse"` as deferred, and rejects other
strings as invalid. This selector is the native adaptation of Python class
objects: `return_as="dense"` means sklearn's `return_as=np.ndarray`. The default
continues to represent unsupported sparse output. `check_feature_matrix[dtype](x: NDArray[dtype],
rows: Int, columns: Int) raises` checks nonnegative expected extents, rank two
and exact shape; it borrows without copying and accepts zero extents. It does
not enforce estimator-specific minimum rows, finite values or dtype policy.

DictVectorizer's child must expose defaults `dtype=float64`, `separator="="`,
`sparse=True`, `sort=True`, with explicit `sparse=False` required to execute.
Normal fit replaces `vocabulary_` and `feature_names_`; no `n_features_in_` is
invented. Clone copies constructor configuration only. `fit_transform` directly
accumulates dense values (duplicate categories sum), transform ignores unknown
keys, inverse returns nonzero named values, and restrict updates column metadata.
Output is `(samples, vocabulary_size)` even for singleton/zero-feature rows.
Empty transform batches and unfitted transform/inverse/name queries must match
reference errors. Fit, set_params and restrict return mutable estimator borrows.
Typed Params follow the foundation contract. Child parity must cover sort modes,
repeat fit, cloning, parameter changes, dtype conversions and invalid values.

## Image shape interfaces

`_image_contracts.checked_extent_product(a: Int, b: Int) raises -> Int` checks
nonnegative extents and signed-Int overflow before multiplication. It is shape
validation, not a replacement array operation.

`PatchGeometry(height, width, patch_height, patch_width, channels=1)` validates
positive extents, a patch fitting the image, and representable element counts.
`positions() raises -> Int` returns `(height-patch_height+1) *
(width-patch_width+1)`; `output_shape(count) raises -> Shape` checks the output
extent product and returns `(count, patch_height, patch_width)` for one channel,
otherwise `(count, patch_height, patch_width, channels)`. Zero selected patches
are valid. Fields are metadata; callers must preserve constructor invariants.

Input images use `(height,width[,channels])`; batches use
`(samples,height,width[,channels])`. Patch order is image-major, then top-to-bottom
and left-to-right. Channel is the innermost axis. Reconstruction returns the
requested image shape and reference Float64 accumulation/output; overlap counts
supply divisors. Raw extraction preserves input dtype; PatchExtractor allocates
Float64 as in the reference. Rank changes remain in NDArray shape, not a flattened
2-D feature matrix. Geometry validation is a private safe precondition, not a
claim of complete sklearn input validation (including its degenerate quirks).

PatchExtractor's constructor defaults remain `patch_size=None`,
`max_patches=None`, `random_state=None`. Fit is stateless and transform does not
require training. None patch_size derives each extent by integer division by 10;
max_patches is a typed None/integer/fraction selection, keeping reference bounds
and sampling semantics. The child owns selection validation and reconstruction
behavior for partial patch inputs, not this geometry helper.

`GridGeometry(n_x, n_y, n_z=1)` validates nonnegative dimensions/product;
`vertices() raises -> Int` returns their product and `dense_shape(selected)
raises -> Shape` validates `0 <= selected <= vertices` and selected squared,
returning `(selected, selected)`. Zero vertices/masks are representable. It does
not validate a mask or compute any edges. C-order vertex index is
`(x*n_y+y)*n_z+z`; a mask compacts retained vertices in that order.

Dense graph engines construct symmetric axial-neighbor adjacency directly.
Grid edge/diagonal values are one; image edge values are absolute pixel
differences and diagonal values are the pixel values. Missing edges are zero.
Image rank 2/3 is spatial (not the channel interpretation of patch extraction).
Masks retain logical shape/order and isolated vertices. Public graph defaults
still require sparse output and must error unless callers select the native
typed equivalent of `return_as=np.ndarray`. Grid defaults to integer dtype;
image graph defaults to its input dtype. Dense memory is O(selected_vertices²),
so check element and byte size before allocation in the engine. The image engine constructs dense graphs directly; no CSR assembly or
sparse conversion is available.

## Deferred extension contracts

`_text_contracts.TextAnalyzer.analyze(self, document: String) raises ->
List[String]` consumes one decoded document and returns owned ordered tokens.
Future adapters handle decoding/file streams before this boundary. Callable
analyzers bypass built-in preprocessing/tokenization; word analyzers preprocess,
tokenize, filter stop words then construct ngrams. Character/char_wb analyzers
need separate Unicode/whitespace rules. `check_ngram_range(min_n, max_n)` checks
`1 <= min_n <= max_n`; no ngrams are generated here.

`DeterministicHasher.hash_utf8(self, key: String, seed: UInt32 = 0) raises ->
Int32` reserves signed MurmurHash3 x86-32 compatibility over exact UTF-8 bytes.
An implementation must preserve reference minimum-int handling, column modulus,
alternate sign and collision summation. This trait supplies no implementation;
Mojo's generic dictionary hash is not a substitute. The dict_hash owner supplies
the future hash kernel, while text consumes it. CSR assembly is explicitly
unavailable until a later sparse owner and user-approved scope change.
TF-IDF owns weighting/normalization and idf_ in its later session, consumes the
same vocabulary column order and cannot introduce new storage. Deferred traits
have raising test probes only; they advertise signatures, not capabilities.

## Dependency assembly and extension

`DEPENDENCIES.json` declares live source paths, module owners, exports and
license assets. Build gates use the current Moxie `shared/assembly` launcher
and read current Moxie and family sources on every run. Source changes need no
repinning. Generated assembly receipts record the consumed bytes. Toolchain
locks and the scikit-learn oracle revision remain fixed independently.
The family build helper validates NuMojo's declared revision and file hashes
before assembly; the live-source policy applies to Moxie and family sources.

Run `pixi install --locked` with project-local `PIXI_CACHE_DIR`, then
`pixi run build` and `pixi run test`. Only `builds/deps` is supplied as the Moxie
import root. Generated initializers come from explicit export manifests.
Update family implementations and conformance probes when live Moxie contracts
change; do not restore old signatures or cached Moxie snapshots.

Algorithm projects declare the transitive sources they consume and retain
explicit module ownership. Never choose competing modules by include-path
order. Identical inherited baseline files may be deduplicated only with the
assembler's verified baseline metadata. Ownership changes require explicit
module/from/to/reason overrides. Global helpers stay in Moxie; family-specific
helpers belong here.

The two isolated fixtures under `tests/consumers/a` and `b` each compose both
independent `_probe_a` and `_probe_b` contributions into one family namespace.
They exchange Float32/Float64 NDArray buffers, use the same vocabulary/validation
helpers and current dense kernel, and prove alias versus snapshot ownership.
The root gate compiles/runs both using independent output/cache/temp directories.
Fixtures are not future algorithm projects. Tests verify shared source hashes,
module/export conflict rejection, missing-source failure without output replacement,
unavailable sparse imports and rejected deferred constructors and all inherited contract probes.
