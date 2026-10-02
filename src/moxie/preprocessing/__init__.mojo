"""Contracts for the preprocessing family."""
from ._base import (
    Preprocessor,
    InvertiblePreprocessor,
    OnlinePreprocessor,
    TargetPreprocessor,
)

# Public scikit-learn API, grouped by native private module.
from ._scalers import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    scale,
    minmax_scale,
    maxabs_scale,
    robust_scale,
)
from ._simple import (
    Normalizer,
    Binarizer,
    KernelCenterer,
    normalize,
    binarize,
    add_dummy_feature,
)
from ._quantiles import (
    QuantileTransformer,
    PowerTransformer,
    quantile_transform,
    power_transform,
)
from ._label import (
    LabelEncoder,
    LabelBinarizer,
    MultiLabelBinarizer,
    label_binarize,
)
from ._encoders import (
    OneHotEncoder,
    OrdinalEncoder,
)
from ._polynomial import (
    PolynomialFeatures,
    SplineTransformer,
)
from ._function_transformer import (
    FunctionTransformer,
)
from ._target_encoder import (
    TargetEncoder,
)
from ._discretization import (
    KBinsDiscretizer,
)
