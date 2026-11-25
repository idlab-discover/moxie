# Dense TF-IDF

```mojo
from moxie.feature_extraction.text import TfidfTransformer
from numojo import Shape, zeros

def main() raises:
    var counts = zeros[DType.int64](Shape(2, 3))
    counts.unsafe_set(0, 2)
    counts.unsafe_set(4, 1)
    var tfidf = TfidfTransformer[DType.int64]()
    var weighted = tfidf.fit_transform(counts)
    var idf = tfidf.idf_()
```

`TfidfTransformer[dtype]` takes a rank-two NuMojo `NDArray[dtype]` with
documents in rows and terms in columns. Float32 and Float64 retain their
dtype. Integer and Boolean counts produce Float64 weights. The input dtype is
fixed for each estimator instance and defaults to Float64. The result is an
owned dense matrix. `transform(X, copy=False)` also leaves the borrowed input
unchanged, as in scikit-learn's dense-input path. Input views may have positive
or negative strides.

The constructor defaults are `norm="l2"`, `use_idf=True`, `smooth_idf=True`
and `sublinear_tf=False`. `norm` accepts `"l1"`, `"l2"` or `None`. With
sublinear weighting, each nonzero count becomes `1 + log(count)`. Smoothing
adds one document to both the corpus size and each term's document frequency.
With smoothing disabled, an unseen feature's IDF is positive infinity.
Zero entries stay zero, even in those columns. As in scikit-learn, normalized
transforms reject nonfinite weighted values. With `norm=None`, those values
can propagate to the result.

`fit` and `set_params` return mutable borrows. `get_params` returns a
`TfidfTransformerParams` record. Pass a record or named keyword updates to
`set_params`. Clone copies configuration into an unfitted transformer.
Successful refits replace learned state, including clearing IDF when
`use_idf=False`. `idf_()` returns an independent rank-one array and raises
when the transformer has no learned IDF. `n_features_in_()` returns the fitted
width. NuMojo arrays carry no column labels. `get_feature_names_out()`
generates `x0`, `x1`, and so on, or returns the supplied `List[String]` after
checking its length.

The TF-IDF loop traverses NuMojo strides directly and allocates only the
result during transform. The weighting and row-norm accumulation run in one
pass, followed by row scaling when requested. Numax's array logarithms and
NuMojo's elementwise logarithms would add whole-matrix buffers and require
masking implicit zeros. Numax `core.libm.log` falls back to the standard
library for Float32. The implementation uses platform libm `log` and `logf`
for both dtypes. The standard dense normalizer suppresses tiny norms with an
epsilon guard. TF-IDF follows the reference CSR normalizer's exact zero check
and retains Float32 rounding of squared values before the Float64 sum.

## Validation

From the Moxie root with the locked Mojo and Python active:

```sh
python shared/assembly/tools/assemble_family.py \
  --consumer families/feature_extraction/algorithms/text \
  --lock DEPENDENCIES.json \
  --output families/feature_extraction/algorithms/text/builds/deps
python families/feature_extraction/algorithms/text/tests/parity.py
```

The oracle executes the local pinned scikit-learn TF-IDF and normalization
bodies. Python tooling uses the locked sklearn wheel's compiled CSR row
normalizers. The tests allow Float32 relative error of 2e-6 and Float64
relative error of 3e-15 for libm versus NumPy rounding and normalization.
Zero, infinity and NaN are checked separately. The family contract test
`tests/test_tfidf.mojo` covers lifecycle, input validation, array ownership,
strided input and `TransformerPipeline` composition.
