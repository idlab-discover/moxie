"""Dense feature extraction and explicit deferred declarations."""
from ._base import FeatureExtractor
from ._dict_vectorizer import DictVectorizer
from ._api import FeatureHasher
from . import image, text
from .image import grid_to_graph, img_to_graph
