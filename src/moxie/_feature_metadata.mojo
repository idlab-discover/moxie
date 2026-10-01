"""Optional feature metadata contracts shared by native composition engines."""


trait FeatureNamesOut:
    """None selects learned/default names; an explicit empty list stays empty."""
    def get_feature_names_out(self, input_features: Optional[List[String]] = None) raises -> List[String]:
        ...


trait InputFeatureCount:
    """Read learned input width without prescribing attribute storage."""
    def get_n_features_in(self) raises -> Int: ...
