from search_support import Adapter, data, grid
from moxie.dummy import DummyRegressor as _DummyRegressor
from moxie.model_selection import GridSearchCV
comptime DummyRegressor = _DummyRegressor[]  # Bind default parameters to the concrete estimator.

def main() raises:
    var arrays = data()
    var search = GridSearchCV[Adapter](DummyRegressor(), grid([1.0]), Adapter())
    _ = search.predict_proba(arrays[0])
