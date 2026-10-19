from moxie.cross_decomposition._contracts import TwoBlockEngine


# The numerical implementation is required, not an optional default.
struct IncompleteEngine(TwoBlockEngine):
    pass


def main():
    var engine = IncompleteEngine()
