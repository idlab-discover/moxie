"""Opaque native values for deferred public API declarations."""
struct APIValue(Copyable, Movable):
    pass


@fieldwise_init
struct APIExpressionDefault(Copyable, Movable):
    """Source identity of a deferred Python callable, type, or sentinel default."""
    var expression: String
