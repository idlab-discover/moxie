"""Impute missing text and keep optional indicators Boolean."""
from moxie.impute import StringMatrix, StringSimpleImputer

def main() raises:
    var cells: List[Optional[String]] = ["blue", "small", None, "large", "red", None]
    var X = StringMatrix(cells, 3, 2)
    var model = StringSimpleImputer(strategy="most_frequent", add_indicator=True)
    var result = model.fit_transform(X)
    print(result.values.get(1, 0).value())  # blue wins the tie
    print(result.values.get(2, 1).value())  # large wins the tie
    print(result.indicators)
    print(model.get_feature_names_out())
