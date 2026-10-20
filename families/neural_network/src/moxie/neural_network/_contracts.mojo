"""Extension seams implemented by future owners, with no default algorithms."""
from moxie._random import RandomState


trait MLPKernel:
    """Associated types must use the layouts in docs/plumbing.md."""
    comptime Matrix: Copyable
    comptime Tensors: Copyable
    comptime Workspace: Copyable

    def initialize(self, mut params: Self.Tensors, mut rng: RandomState, activation: String) raises: ...
    def forward(self, X: Self.Matrix, params: Self.Tensors, hidden: String, output: String) raises -> Self.Matrix: ...
    def loss_gradient(
        self, X: Self.Matrix, y: Self.Matrix, weights: Optional[Self.Matrix],
        params: Self.Tensors, mut gradients: Self.Tensors, mut work: Self.Workspace,
        hidden: String, output: String, loss: String, alpha: Float64,
    ) raises -> Float64: ...


trait MLPOptimizer:
    comptime Tensors: Copyable

    def update_params(mut self, mut params: Self.Tensors, gradients: Self.Tensors) raises: ...
    def iteration_ends(mut self, time_step: Int) raises: ...
    def trigger_stopping(mut self, message: String, verbose: Bool) raises -> Bool: ...


trait LBFGSObjective:
    """Separate full-batch objective, not an incremental optimizer."""
    comptime Vector: Copyable

    def value_gradient(mut self, packed: Self.Vector, mut gradient: Self.Vector) raises -> Float64: ...


trait RBMSampling:
    """Optional RBM capability alongside the preserved RBMTransformer trait."""
    comptime Input: Copyable
    comptime Samples: Copyable

    def gibbs(mut self, X: Self.Input) raises -> Self.Samples: ...
    def partial_fit(mut self, X: Self.Input, y: Optional[Self.Input] = None) raises: ...
