"""A caller cannot pair an adapter with a different kernel type."""
from moxie.kernel_ridge import KernelRidge
from moxie.kernel_ridge_serialization import KernelRidgeSchema
from moxie._kernel_ridge._estimator import NoKernel
from callable_kernel import AffineKernelAdapter


def main() raises:
    _ = KernelRidgeSchema[DType.float64, NoKernel, AffineKernelAdapter].encode(
        KernelRidge()
    )
