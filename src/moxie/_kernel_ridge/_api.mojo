"""KernelRidge-specific native API values; no global routing engine."""
from std.utils import Variant


comptime KernelParameterValue = Variant[Float64, Int, Bool, String]
comptime KernelParameters = Dict[String, KernelParameterValue]


struct ParameterUpdate[T: Copyable & Deinitable](Copyable, Movable):
    """Distinguish an omitted keyword from an explicit None."""

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
    def __init__(out self, value: NoneType._mlir_type):
        self.supplied = True
        self.value = None


comptime WeightRequest = Optional[Variant[Bool, String]]


def validate_request(request: WeightRequest) raises:
    if request and request.value().isa[String]():
        var name = request.value()[String].copy()
        if name == "$UNCHANGED$":
            return
        if name.byte_length() == 0:
            raise Error("ValueError: metadata alias must be an identifier")
        for i in range(name.byte_length()):
            var c = name[byte=i]
            if not (
                c >= "a"
                and c <= "z"
                or c >= "A"
                and c <= "Z"
                or c == "_"
                or i > 0
                and c >= "0"
                and c <= "9"
            ):
                raise Error(
                    "ValueError: metadata alias must be an ASCII identifier"
                )


def unchanged(request: WeightRequest) -> Bool:
    return (
        Bool(request)
        and request.value().isa[String]()
        and request.value()[String] == "$UNCHANGED$"
    )


struct KernelRidgeMethodRequest(Copyable, Movable):
    var sample_weight: WeightRequest

    def __init__(out self):
        self.sample_weight = None

    def route[
        T: Copyable & Deinitable
    ](self, metadata: Dict[String, T]) raises -> Optional[T]:
        """Select sample_weight for a native parent; None rejects supplied data.
        """
        if not self.sample_weight:
            if "sample_weight" in metadata:
                raise Error(
                    "UnsetMetadataPassedError: sample_weight was not requested"
                )
            return None
        var request = self.sample_weight.value().copy()
        if request.isa[Bool]():
            if not request[Bool]:
                return None
            if "sample_weight" in metadata:
                return metadata["sample_weight"].copy()
        elif request[String] in metadata:
            return metadata[request[String]].copy()
        return None


@fieldwise_init
struct KernelRidgeMetadataRequest(Copyable, Movable):
    var fit: KernelRidgeMethodRequest
    var score: KernelRidgeMethodRequest
