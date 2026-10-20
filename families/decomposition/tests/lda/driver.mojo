"""Driver: f32|f64 X.txt k method max_iter batch_size evaluate_every seed doc_prior(-1 None) word_prior(-1 None)
learning_decay learning_offset max_doc_update_iter mean_change_tol perp_tol total_samples mode(fit|partial)."""
from std.sys import argv
from moxie.decomposition import LatentDirichletAllocation
from decomposition_testing import emit, emit_scalar, read_matrix


def evaluate[dtype: DType]() raises:
    var args = argv()
    var x = read_matrix[dtype](String(args[2]))
    var doc_prior: Optional[Float64] = None
    if Float64(String(args[9])) >= 0: doc_prior = Float64(String(args[9]))
    var word_prior: Optional[Float64] = None
    if Float64(String(args[10])) >= 0: word_prior = Float64(String(args[10]))
    var model = LatentDirichletAllocation[dtype](Int(args[3]), learning_method=String(args[4]), max_iter=Int(args[5]),
        batch_size=Int(args[6]), evaluate_every=Int(args[7]), random_state=Int(args[8]), doc_topic_prior=doc_prior,
        topic_word_prior=word_prior, learning_decay=Float64(String(args[11])), learning_offset=Float64(String(args[12])),
        max_doc_update_iter=Int(args[13]), mean_change_tol=Float64(String(args[14])), perp_tol=Float64(String(args[15])),
        total_samples=Float64(String(args[16])))
    if String(args[17]) == "partial":
        model.partial_fit(x)
        model.partial_fit(x)
    else:
        model.fit(x)
        emit_scalar("bound", model.bound_)
        emit_scalar("n_iter", Float64(model.n_iter_))
    emit("components", model.components_.value())
    emit("exp_dirichlet", model.exp_dirichlet_component_.value())
    emit_scalar("n_batch_iter", Float64(model.n_batch_iter_))
    emit("transform", model.transform(x))
    emit("unnormalized", model.transform(x, normalize=False))
    emit_scalar("score", model.score(x))
    emit_scalar("perplexity", model.perplexity(x))
    emit_scalar("perplexity_sub", model.perplexity(x, sub_sampling=True))


def main() raises:
    if argv()[1] == "f32": evaluate[DType.float32]()
    else: evaluate[DType.float64]()
