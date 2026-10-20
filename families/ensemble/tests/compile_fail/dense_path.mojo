from moxie.ensemble._engine_api import DenseForestInspection

def forbidden[T: DenseForestInspection](model: T, X: T.Input) raises:
    _ = model.decision_path(X)

def main():
    pass
