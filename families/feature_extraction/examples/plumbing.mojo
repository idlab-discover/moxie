"""Column metadata and patch shapes; no estimator implementation."""
from moxie.feature_extraction._mapping import category_key, OrderedVocabulary
from moxie.feature_extraction._image_contracts import PatchGeometry
from moxie.feature_extraction._validation import require_dense_output


def main() raises:
    require_dense_output(False)
    var vocabulary = OrderedVocabulary[String]([category_key("city", "Paris"), "temperature"])
    print("city column:", vocabulary.column("city=Paris"))
    var geometry = PatchGeometry(8, 8, 3, 3)
    print("patch positions:", geometry.positions())
