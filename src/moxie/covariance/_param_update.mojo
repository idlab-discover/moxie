"""Nullable keyword updates: omission preserves, explicit None clears."""


struct NullableUpdate[T: Copyable & Deinitable](Copyable, Movable):
    var supplied: Bool
    var value: Optional[Self.T]

    def __init__(out self):
        self.supplied = False
        self.value = None

    @implicit
    def __init__(out self, var value: Self.T):
        self.supplied = True
        self.value = value^

    @implicit
    def __init__(out self, var value: Optional[Self.T]):
        self.supplied = True
        self.value = value^

    @implicit
    def __init__(out self, none: NoneType._mlir_type):
        self.supplied = True
        self.value = None
