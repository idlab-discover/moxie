from moxie.kernel_ridge import KernelRidge, ScalarOrVector


def escape[origin: MutOrigin](
    ref[origin] model: KernelRidge[]
) raises -> ref[model] KernelRidge[]:
    var local = KernelRidge()
    return local.set_params(alpha=ScalarOrVector(2.0))


def main() raises:
    var model = KernelRidge()
    _ = escape(model)
