from moxie.pipeline import union_step
from moxie.dummy import DummyRegressor

def main() raises:
    _ = union_step("wrong", DummyRegressor())
