# Decomposition interfaces

Follow the [current Moxie interface contract](../../../docs/interfaces.md).
Typed parameter records, static dtypes and explicit unsupported modes are
specified in the [family API](api-surface.md) and individual estimator guides.
Public fit, partial_fit and set_params return a mutable reference to the original
estimator. Historical scaffold interfaces remain in the standalone Git history.
