"""Dense kernel ridge regression over NuMojo arrays."""
from ._kernel_ridge._estimator import (
    KernelRidge,
    KernelRidgeParams,
    KernelRidgeRegressor,
    ScalarOrVector,
)
from ._kernel_ridge._gp_kernel import GaussianProcessKernel, TypedGaussianProcessKernel
from ._kernel_ridge._kernels import MatrixKernel, RowKernel

from ._kernel_ridge._promotion import FloatingArray, FittedArray

from ._kernel_ridge._api import (
    KernelRidgeMetadataRequest,
    KernelRidgeMethodRequest,
    WeightRequest,
    KernelParameterValue,
    KernelParameters,
)
