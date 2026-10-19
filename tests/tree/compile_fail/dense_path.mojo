from moxie.tree._dense_api import DenseTreeInspection

struct DenseProbe(DenseTreeInspection):
    comptime Params = Int
    comptime Input = Int
    comptime Leaves = Int
    var marker: Int
    def __init__(out self):
        self.marker = 0
    def get_params(self, deep: Bool = True) raises -> Int:
        return 0
    def set_params(mut self, params: Int) raises:
        self.marker = params
    def __sklearn_clone__(self) raises -> Self:
        return Self()
    def __sklearn_is_fitted__(self) -> Bool:
        return False
    def apply(self, X: Int, check_input: Bool = True) raises -> Int:
        raise Error("No engine")

def main() raises:
    var probe = DenseProbe()
    _ = probe.decision_path(1)
