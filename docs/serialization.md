# Native model files (experimental)

`moxie.serialization` owns the shared binary container and schema adapter
contract. Families own estimator identifiers, state schemas, reconstruction and
semantic validation. This feature is under development on
`feature/native-model-serialization`; format version 1 is not a stable release
promise. It uses neither pickle nor safetensors.

A family implements `ModelSchema`, with an associated `Model` type and static
`encode(model) -> ModelArchive` / `decode(archive) -> Model` methods. Decode
must check the exact estimator identifier and schema version, validate all state,
and construct an independent estimator without calling fit. Constructor settings
and learned state are distinct: settings changed after fitting must not overwrite
learned state during reconstruction. Reject unknown fields/arrays unless the
family explicitly defines forward-compatible behavior.

```mojo
from moxie.serialization import save_model, load_model
from moxie.dummy_serialization import DummyRegressorSchema

# Requires the dummy family's assembled namespace.
save_model[DummyRegressorSchema](model, "baseline.moxie")
var restored = load_model[DummyRegressorSchema]("baseline.moxie")
var predictions = restored.predict(X)
```

Loading requires the caller to choose a schema at compile time. File contents
cannot import modules, name executable functions, or select arbitrary classes.
There is no automatic heterogeneous estimator registry. Pipeline owns its future
composition schema; fitted pipeline persistence is not implemented here.

## Shared API

- `ModelArchive(type_id, schema_version)` holds opaque manifest bytes and named
  `ArrayPayload` records. `expect` checks family identity/version; `validate`
  checks structural consistency and uniqueness.
- `BinaryWriter` / `BinaryReader` encode/decode length-prefixed UTF-8 strings,
  byte strings, signed 64-bit integers, unsigned 64-bit integers, strict boolean
  values, and bit-exact 32/64-bit scalars. Reader methods check bounds before
  access. Call `finish()` after reading a schema to reject surplus fields.
- `archive.add_array[dtype](name, shape, values)` accepts spans in logical C
  order. `archive.array(name).values[dtype]()` returns an independent scalar
  list with a checked dtype. These are wire buffers, not a replacement numerical
  array class. Storage adapters must traverse logical values, including strides.
- `dumps` / `loads` handle memory; `save` / `load` handle one file;
  `save_model[Schema]` / `load_model[Schema]` also invoke the family adapter.
- Supported payload dtypes: Float32, Float64, Int32, Int64, UInt32, UInt64.
  Payload scalars preserve every bit, including signed zero and NaN payloads.
  Strings preserve UTF-8 bytes, including embedded NULs. Labels and optional
  values use the family's manifest schema.

## Optional compression

Import `SnappyCodec` from `moxie.serialization_snappy` and pass it as the second
parameter to `save_model[Schema, SnappyCodec]` / `load_model[Schema, SnappyCodec]`,
or the first parameter to `save`, `load`, `dumps`, and `loads`. A Snappy-enabled
loader also accepts raw payloads. A raw-only loader explicitly rejects compressed
payloads. Compression is per array, opt-in, and used only when it reduces bytes.
The manifest stays uncompressed. No speed or universal size benefit is promised.

`SNAPPY.lock.json` pins mojo-snappy 0.2.0 source and license hashes for Mojo 1.0.0.
`python tools/prepare_snappy.py` recovers those exact Git objects into
`builds/snappy`; verified cached bytes work without the source checkout. Include
`-I builds/snappy/src` when importing the codec, precompiling the whole package,
or consuming that precompiled package (Mojo resolves its recorded codec
dependency eagerly). The core `serialization` module can be imported from source
without Snappy. No third-party codec source is copied into tracked Moxie library
files.

## Wire format, version 1

All integers and numerical values use little-endian order. No native pointers,
compiler dtype enum ordinals, memory layouts, or process-dependent hashes are
stored. Lengths/counts/versions/flags below are unsigned 64-bit words.

The 32-byte header contains the 8 bytes `MOXIEMDL`, container version `1`, body
byte length, and IEEE CRC-32 of the complete body, zero-extended to 64 bits.
The body contains, in order:

1. UTF-8 estimator ID (byte length then bytes), positive family schema version,
   opaque manifest (byte length then bytes), and array count.
2. For each array: UTF-8 name, dtype code, rank, dimensions, decoded byte length,
   codec ID, stored byte length, and stored bytes.

Dtype codes are 1=Float32, 2=Float64, 3=Int32, 4=Int64, 5=UInt32, 6=UInt64.
Codec IDs are 0=raw, 1=mojo-snappy raw block. Rank-zero arrays contain one scalar;
any zero dimension makes an empty array. Dimensions are bounded to 2^31−1.
Array names are nonempty and unique. Estimator IDs and array names are at most
4096 UTF-8 bytes. No trailing data is accepted. Versions and unknown dtypes or
codecs fail explicitly. Schema migration requires an explicit family adapter.

## Bounds and limitations

Default `LoadLimits`: 256 MiB file, 256 MiB aggregate decoded array bytes,
8 MiB manifest, 1024 arrays, rank 32. Pass smaller limits when appropriate.
File reads use bounded chunks. The checksum is checked before parsing the body;
shape products, encoded lengths and aggregate decoded sizes are checked before
allocating decoded payloads. Snappy's advertised size must equal the container's
size before decompression. These are data limits, not an exact process RSS cap:
input, decoded arrays, reconstruction and transient copies may coexist.

CRC detects accidental corruption; it does not authenticate the producer.
Semantic checks belong to family adapters. Saving validates/encodes before
opening the destination, but a failed write can leave a partial file; replacement
is not transactional. Uncompressed files are copied on load. Memory mapping,
streaming array access, arbitrary object graphs, sparse payloads, shared object
identity and external RNG stream snapshots are not implemented.

Run `pixi run build` and `pixi run test`. Tests cover byte-exact round trips,
all supported dtypes, empty arrays, malformed structure with recomputed checksums,
truncation at every offset, corruption at every byte, limits, codec rejection,
Snappy round trips and an independent Python struct/zlib wire oracle. Python is
validation/build tooling only. Dummy owns its real estimator round-trip tests.
