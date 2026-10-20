# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie.feature_extraction._api_types import APIValue
from moxie.feature_extraction._api_types import APIExpressionDefault

struct FeatureHasher(Movable):

    "API scaffold for sklearn.feature_extraction._hash.FeatureHasher. Operations are deferred."

    var n_features: APIValue

    var input_type: APIValue

    var dtype: APIValue

    var alternate_sign: APIValue

    def get_metadata_routing(mut self) raises -> APIValue:
        "Scikit-learn signature: get_metadata_routing(self)."
        raise Error("NotImplementedError: FeatureHasher.get_metadata_routing is an API scaffold")

    def get_params(mut self, deep: Some[Movable] = True) raises -> APIValue:
        "Scikit-learn signature: get_params(self, deep=True)."
        raise Error("NotImplementedError: FeatureHasher.get_params is an API scaffold")

    def set_params(mut self, params: Some[Movable] = None) raises:
        "Scikit-learn signature: set_params(self, **params)."
        raise Error("NotImplementedError: FeatureHasher.set_params is an API scaffold")

    def set_output(mut self, *, transform: Some[Movable] = None) raises:
        "Scikit-learn signature: set_output(self, *, transform=None)."
        raise Error("NotImplementedError: FeatureHasher.set_output is an API scaffold")

    def fit_transform(mut self, X: Some[Movable], y: Some[Movable] = None, fit_params: Some[Movable] = None) raises -> APIValue:
        "Scikit-learn signature: fit_transform(self, X, y=None, **fit_params)."
        raise Error("NotImplementedError: FeatureHasher.fit_transform is an API scaffold")

    def __init__(out self, n_features: Some[Movable] = 1048576, *, input_type: Some[Movable] = "dict", dtype: Some[Movable] = DType.float64, alternate_sign: Some[Movable] = True) raises:
        "Scikit-learn signature: __init__(self, n_features=2 ** 20, *, input_type='dict', dtype=np.float64, alternate_sign=True)."
        raise Error("NotImplementedError: FeatureHasher.__init__ is an API scaffold")

    def fit(mut self, X: Some[Movable] = None, y: Some[Movable] = None) raises:
        "Scikit-learn signature: fit(self, X=None, y=None)."
        raise Error("NotImplementedError: FeatureHasher.fit is an API scaffold")

    def transform(mut self, raw_X: Some[Movable]) raises -> APIValue:
        "Scikit-learn signature: transform(self, raw_X)."
        raise Error("NotImplementedError: FeatureHasher.transform is an API scaffold")

def grid_to_graph(n_x: Some[Movable], n_y: Some[Movable], n_z: Some[Movable] = 1, *, mask: Some[Movable] = None, return_as: Some[Movable] = APIExpressionDefault("sparse.coo_array"), dtype: Some[Movable] = APIExpressionDefault("int")) raises -> APIValue:
    "Scikit-learn signature: grid_to_graph(n_x, n_y, n_z=1, *, mask=None, return_as=sparse.coo_array, dtype=int)."
    raise Error("NotImplementedError: grid_to_graph is an API scaffold")

def img_to_graph(img: Some[Movable], *, mask: Some[Movable] = None, return_as: Some[Movable] = APIExpressionDefault("sparse.coo_array"), dtype: Some[Movable] = None) raises -> APIValue:
    "Scikit-learn signature: img_to_graph(img, *, mask=None, return_as=sparse.coo_array, dtype=None)."
    raise Error("NotImplementedError: img_to_graph is an API scaffold")
