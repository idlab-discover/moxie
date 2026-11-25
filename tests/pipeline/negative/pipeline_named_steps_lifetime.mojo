from moxie.pipeline import make_pipeline
from moxie.preprocessing import StandardScaler


def main() raises:
    var pipe = make_pipeline(StandardScaler(), StandardScaler())
    var named = pipe.named_steps()
    var moved = pipe^
    print(len(named), len(moved))
