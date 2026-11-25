"""Native diagnostics recorded by estimators during a successful fit."""


@fieldwise_init
struct FitWarning(Copyable, Movable):
    """A warning category name and its message."""

    var category: String
    var message: String
