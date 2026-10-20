# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""make_pipeline/make_union: typed constructors with sklearn's automatic step names."""
from std.reflection import reflect
from moxie.base import TransformerMixin, SupervisedEstimator, Predictor
from moxie._pipeline_sequential import Pipeline
from moxie._pipeline_union import FeatureUnion, UnionStep, UnionPair, union_step


def _name_estimators(var names: List[String]) -> List[String]:
    """sklearn's _name_estimators on lowercased type names: duplicates get -1, -2, ..."""
    for i in range(len(names)):
        names[i] = names[i].lower()
    var counts = List[Int]()
    for i in range(len(names)):
        var count = 0
        for j in range(len(names)):
            if names[j] == names[i]:
                count += 1
        counts.append(count)
    var seen = List[Int]()
    for i in range(len(names)):
        var index = 1
        for j in range(i):
            if names[j] == names[i]:
                index += 1
        seen.append(index)
    for i in range(len(names)):
        if counts[i] > 1:
            names[i] += "-" + String(seen[i])
    return names^


def _type_name[T: AnyType]() -> String:
    comptime info = reflect[T]
    return String(info.base_name())


def make_pipeline[P: TransformerMixin & Deinitable, E: SupervisedEstimator & Predictor & Deinitable](
    var transformer: P, var estimator: E,
) raises -> Pipeline[P, E] where conforms_to(P.Params, Deinitable) and conforms_to(E.Params, Deinitable):
    """Scikit-learn: make_pipeline(*steps, memory=None, transform_input=('X_val',), verbose=False).

    Names are the lowercased type names, numbered on collision, as in sklearn.
    Two steps map to the typed two-stage Pipeline; nest a Pipeline for more.
    `memory`, `transform_input` and `verbose` are not implemented by Pipeline.
    """
    var names = _name_estimators([_type_name[P](), _type_name[E]()])
    return Pipeline(transformer^, estimator^, transformer_name=names[0], estimator_name=names[1])


def make_union[A: TransformerMixin & Deinitable, B: TransformerMixin & Deinitable, //, dtype: DType = DType.float64](
    var first: A, var second: B, *, n_jobs: Optional[Int] = None, verbose: Bool = False,
    verbose_feature_names_out: Bool = True,
) raises -> FeatureUnion[UnionPair[UnionStep[dtype, A], UnionStep[dtype, B]]] where conforms_to(A.Params, Deinitable) and conforms_to(B.Params, Deinitable):
    """Scikit-learn: make_union(*transformers, n_jobs=None, verbose=False, verbose_feature_names_out=True)."""
    var names = _name_estimators([_type_name[A](), _type_name[B]()])
    return FeatureUnion(UnionPair(union_step[dtype](names[0], first^), union_step[dtype](names[1], second^)),
                        n_jobs=n_jobs, verbose=verbose, verbose_feature_names_out=verbose_feature_names_out)


def make_union[
    A: TransformerMixin & Deinitable, B: TransformerMixin & Deinitable, C: TransformerMixin & Deinitable, //,
    dtype: DType = DType.float64,
](
    var first: A, var second: B, var third: C, *, n_jobs: Optional[Int] = None, verbose: Bool = False,
    verbose_feature_names_out: Bool = True,
) raises -> FeatureUnion[UnionPair[UnionStep[dtype, A], UnionPair[UnionStep[dtype, B], UnionStep[dtype, C]]]] where conforms_to(A.Params, Deinitable) and conforms_to(B.Params, Deinitable) and conforms_to(C.Params, Deinitable):
    """Three-transformer make_union; the nested pairs form one flat branch list."""
    var names = _name_estimators([_type_name[A](), _type_name[B](), _type_name[C]()])
    var tail = UnionPair(union_step[dtype](names[1], second^), union_step[dtype](names[2], third^))
    return FeatureUnion(UnionPair(union_step[dtype](names[0], first^), tail^),
                        n_jobs=n_jobs, verbose=verbose, verbose_feature_names_out=verbose_feature_names_out)
