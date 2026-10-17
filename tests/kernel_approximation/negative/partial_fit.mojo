from moxie.kernel_approximation import Nystroem
def main() raises:
    var m = Nystroem(random_state=1)
    m.partial_fit(1)
