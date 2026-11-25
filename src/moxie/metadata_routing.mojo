"""Typed metadata requests for sample-weight consumers."""

from std.utils import Variant


comptime _UNCHANGED = "$UNCHANGED$"
comptime _WEIGHT = "sample_weight"
comptime SampleWeightRequestValue = Variant[Bool, String, NoneType]


struct SampleWeightUpdate(Copyable, Movable):
    """Setter input. The default retains the previous request."""

    var _value: SampleWeightRequestValue

    def __init__(out self):
        self._value = String(_UNCHANGED)

    @implicit
    def __init__(out self, value: Bool):
        self._value = value

    @implicit
    def __init__(out self, var alias_name: String):
        self._value = alias_name^

    @implicit
    def __init__(out self, value: NoneType._mlir_type):
        self._value = NoneType()


struct SampleWeightRequest(Copyable, Movable):
    """A method's validated request for sample weights."""

    var _value: SampleWeightRequestValue

    def __init__(out self):
        self._value = NoneType()

    def value(self) -> SampleWeightRequestValue:
        """Return the stored request as an independent value."""
        return self._value.copy()

    def set(mut self, request: SampleWeightUpdate) raises:
        if request._value.isa[String]():
            ref name = request._value[String]
            if name == _UNCHANGED:
                return
            if name == _WEIGHT:
                self._value = True
                return
            if name.byte_length() == 0:
                raise Error("ValueError: metadata alias must be an ASCII identifier")
            for i in range(name.byte_length()):
                var c = name[byte=i]
                if not (
                    (c >= "a" and c <= "z")
                    or (c >= "A" and c <= "Z")
                    or c == "_"
                    or (i > 0 and c >= "0" and c <= "9")
                ):
                    raise Error("ValueError: metadata alias must be an ASCII identifier")
        self._value = request._value.copy()

    def requested_name(self) -> Optional[String]:
        """Return the name a parent should use when the request consumes weights."""
        if self._value.isa[String]():
            return self._value[String].copy()
        elif self._value.isa[Bool]() and self._value[Bool]:
            return String(_WEIGHT)
        return None

    def route[T: Copyable & Deinitable](
        self, metadata: Dict[String, T]
    ) raises -> Optional[T]:
        """Resolve one method's weights from a parent's validated metadata."""
        if self._value.isa[NoneType]() and _WEIGHT in metadata:
            raise Error("UnsetMetadataPassedError: sample_weight was not requested")
        var name = self.requested_name()
        if name and name.value() in metadata:
            return metadata[name.value()].copy()
        return None


@fieldwise_init
struct FitScoreMetadataRequest(Copyable, Movable):
    """Independent requests for a supervised estimator's fit and score."""

    var fit: SampleWeightRequest
    var score: SampleWeightRequest
