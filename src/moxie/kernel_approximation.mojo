"""Dense kernel feature maps. See docs/kernel_approximation/README.md."""
from moxie._kernel_approximation_contracts import KernelFeatureMap
from moxie._kernel_additive import AdditiveChi2Sampler
from moxie._kernel_fourier import RBFSampler
from moxie._kernel_fourier import SkewedChi2Sampler
from moxie._kernel_nystroem import Nystroem
from moxie._kernel_nystroem import NystroemKernelParams
from moxie._kernel_nystroem import NoCallableKernel
from moxie._kernel_polynomial import PolynomialCountSketch
