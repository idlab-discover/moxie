# Feature extraction

Moxie owns the dense dictionary and image implementations under this directory.
Public imports remain `moxie.feature_extraction` and
`moxie.feature_extraction.image`. Python is used only for validation and tooling.

Implemented: dense `DictVectorizer`, `PatchExtractor`, `extract_patches_2d`,
`reconstruct_from_patches_2d`, `img_to_graph` and `grid_to_graph`.
Select `sparse=False` for dictionary vectorization and `return_as="dense"`
for graph output. The reference sparse defaults still raise.

FeatureHasher, text vectorizers and TF-IDF are raising declarations, not
implementations. Complex and extended-precision dictionary dtypes remain
unfinished. See [scope](docs/scope.md), [dictionary API](algorithms/dict_hash/docs/api.md)
and [image API](algorithms/image/docs/api.md) for supported behavior and adaptations.

Run from the Moxie root:

```sh
pixi run feature-extraction-build
pixi run feature-extraction-test
pixi run feature-extraction-parity
pixi run feature-extraction-start
pixi run feature-extraction-independence
```

Test includes builds, native contracts, six isolated consumers, declaration
rejection and assembly failure checks. Parity compares the real-valued dictionary
and image algorithms to AST-selected bodies from the local scikit-learn checkout.
It does not validate Python decorator machinery or establish full sklearn parity.
Generated numerical cases compile in batches of at most 28 functions under
Moxie's shared compiler resource limits; all cases still execute.
Parity assembles fresh inputs. Run build or test before start.

Side-by-side notebooks against the official gallery live in
`notebooks/feature_extraction/`: `plot_hashing_vs_dict_vectorizer` (dense
DictVectorizer on token-frequency dicts, inverse/restrict, pipeline),
`plot_image_denoising` (patch extraction, sampling, PatchExtractor and
reconstruction) and `plot_segmentation_toy` (dense `img_to_graph` and
`grid_to_graph`). They also prove that FeatureHasher, the text vectorizers and
the sparse defaults fail clearly. Run them with
`pixi run -e notebooks notebooks notebooks/feature_extraction`.

The root `DEPENDENCIES.json` assembles the public namespace from live Moxie
owners. Local consumer manifests exercise narrower contracts independently.
[Provenance](PROVENANCE.json) records the original Git revision; its history and
ignored artifacts remain in `retired-moxie-feature_extraction`. That checkout is
not a build dependency. Compatible toolchain locks remain unchanged.
