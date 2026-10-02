"""Owned feature-name metadata, independent of numerical storage."""
from moxie.utils.validation import FittedState


struct FeatureNames(Copyable, Movable):
    var state: FittedState
    var names: List[String]

    def __init__(out self):
        self.state = FittedState()
        self.names = List[String]()

    def reset(mut self):
        self.state.reset()
        self.names.clear()

    def record_fit(mut self, n_samples: Int, n_features: Int,
                   names: Optional[List[String]] = None) raises:
        var next_names = List[String]()
        if names:
            if len(names.value()) != n_features:
                raise Error("ValueError: feature names have wrong length")
            next_names = names.value().copy()
        var next_state = FittedState()
        next_state.record_fit(n_samples, n_features)
        self.state = next_state^
        self.names = next_names^

    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        if not self.state.is_fitted():
            raise Error("NotFittedError: feature names require fit")
        var count = self.state.n_features_in_.value()
        if input_features:
            var supplied = input_features.value().copy()
            if len(supplied) != count:
                raise Error("ValueError: feature names have wrong length")
            if len(self.names) != 0:
                for i in range(count):
                    if supplied[i] != self.names[i]:
                        raise Error("ValueError: feature names differ from fit")
            return supplied^
        if len(self.names) != 0:
            return self.names.copy()
        var generated = List[String]()
        for i in range(count):
            generated.append("x" + String(i))
        return generated^
