from moxie.kernel_ridge import KernelRidge, ScalarOrVector


def mutate(model: KernelRidge[]) raises:
    _ = model.set_params(alpha=ScalarOrVector(2.0))


def main() raises:
    var model = KernelRidge()
    mutate(model)
