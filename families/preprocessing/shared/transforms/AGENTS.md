## Current family development policy

Read [../../docs/development.md](../../docs/development.md) before development. Preprocessing is integrated
into Moxie. Preserve module ownership, toolchain locks and correctness-reference
revisions; commit changes in the Moxie repository.

# Dense transform engine

This directory owns `moxie.preprocessing._dense_transforms`. Reuse `numojo.NDArray`,
`Shape`, `Item`, Float32/64 storage; `moxie._dense.sum_axis2d` and
`row_squared_norms` retain numerical-owner semantics. Consume family
`_validation`, `_metadata`, `_contracts`, `_base` and foundation `moxie.base`,
`moxie.utils.validation.FittedState` through foundation assembly.
Use `moxie._random.RandomState` if randomness is needed. Keep storage in its shared owner and family helpers in this repository. Sparse
implementation remains deferred. Use mojo-syntax for Mojo.
Run build/test/parity/memory; preserve Mojo 1.0.0/MAX 26.5.0 and contract=off.
