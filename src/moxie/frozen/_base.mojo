"""Interface trait for the frozen family; FrozenEstimator lives in _frozen. See docs/frozen/api.md."""
from moxie.base import BaseEstimator


trait FrozenEstimatorAPI(BaseEstimator):
    pass
