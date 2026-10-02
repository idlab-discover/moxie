# Memory contract

One worker; default aggregate tile budget B=65,536 bytes, configurable from
8 bytes through 8 MiB. Arithmetic kernels use scalar scratch (O(1)); sink paths
allocate one tile with at most B data bytes. NuMojo shape/stride/refcount metadata
adds small fixed allocations. Tiles are logical pieces, not statistical calls.

| Path | Additional data allocation and lifetime |
| --- | --- |
| transform/normalize (copying) | Exactly n*p*d bytes; result transferred to caller |
| in-place/caller-output | No data allocation; borrowed view metadata only for in-place |
| transform_sink/normalize_sink | One reused min(vector length*d,B) tile until return |
| norms_into | None; caller provides n*d (rows) or p*d (features) norm output |
| FeatureNorms | Exactly one p*d state vector, construction through destruction; update/finish/apply allocate no data |

Affine parameters are two caller-owned O(p) vectors (O(n) for row parameters).
Mixed dtype operations promote scalars only. Copying paths allocate logical shape,
never call `.copy()` on views. Input plus a requested whole output must fit RAM.
Norms require a reduction pass and a write pass. Feature streaming requires
complete global norm accumulation followed by replay, as described in
[transforms.md](transforms.md). No past chunk is retained or concatenated.
Weights and missing-mask statistics do not belong to this transform engine;
signed values and NaNs are tested without allocating full-table intermediates.

Run `pixi run memory`. The tracked driver compiles native workloads, measures
isolated per-process peak RSS against identically touched input-only controls,
uses separate controls for caller destinations, audits all engine allocation
sites, and counts NuMojo's TCMalloc entries with Callgrind. Default Massif/DHAT
miss this allocator and are not used as proof. RSS alone is not the gate.

The declared subprocess RAM budget is 512 MiB. Cases include tall 32768x8,
wide 8x32768, large 8192x1024, Float32/64, reversed strides, and a six-element
view into an 8,388,608-element backing allocation. Tiles of 256 and 65536 bytes
exercise both ceilings. Controls touch the complete backing allocations and
caller-owned destinations. The RSS overhead allowance is 12 MiB for process and
allocator noise; the large cases' extra tables are 32/64 MiB. A deliberate full
table temporary must exceed the in-place budget and increase allocator counts.
The source audit rejects new allocation sites and backing-copy operations until
reviewed. Native contracts additionally verify alias identity, preservation,
read-only/overlap failures, tile sizes and whole/stream numerical equality.

Raw measurements are under ignored `benchmarks/memory/`; build flags are
`-O3 --fp-mode=contract=off`, Mojo 1.0.0/MAX 26.5.0. Use
`pixi run python tools/benchmark.py --mode copy` (or baseline/reference) for pyperf
measurements. Setup/compilation/process launch are outside native timed regions;
checksums are inside both native implementations. The reference timing uses the
pinned affine formula as NumPy stages. Runner peers may distort timing; no
speedup claim is made. Public estimator consumers must rerun their own gates.

An initial large-case run (8192x1024, B=65536) measured approximately 52.5 MiB
input-only / 84.9 MiB copying for Float32, and 84.4 / 148.5 MiB for Float64.
In-place and sink peaks remained within about 1 MiB of their input controls;
caller-output peaks matched their input-plus-destination controls. These are
host observations, not portable process-RSS bounds. TCMalloc counted 5 allocations
per logical copied output (data plus metadata), 2 metadata allocations per
in-place call, and 0 for caller-output calls. Sink handles have per-tile metadata
allocation traffic, but only one tile's data remains live. The gate also tests
feature-state streaming and NaN-containing inputs; raw results identify each run.
