"""Cross-validate and tune a real mixed-table pipeline using independent folds."""
from moxie.base import TransformerMixin
from moxie.compose import ColumnEntry, ColumnTable, ColumnTransformer
from moxie.impute import SimpleImputer
from moxie.linear_model import Ridge
from moxie.model_selection import DenseSupervisedAdapter, ErrorPolicy, GridSearchCV, cross_val_score
from moxie.pipeline import Pipeline, make_pipeline
from moxie.preprocessing import OneHotEncoder, StandardScaler
from numojo import Item, Shape, zeros
from parity_io import ResultLine
from std.math import nan
from std.testing import assert_equal, assert_false, assert_raises

from mixed_ridge import table, vector


def apply_alpha[P: TransformerMixin & Deinitable](
    mut model: Pipeline[P, Ridge[]], params: Dict[String, Float64],
) raises:
    var config = model.get_params()
    config.estimator.alpha = params["ridge__alpha"]
    _ = model.set_params(config)


def main() raises:
    var missing = nan[DType.float64]()
    var train = table(
        [10, missing, 30, 40, 20, missing, 65, 52, 79],
        ["red", "blue", "green", "red", "amber", "blue", "red", "amber", "red"],
    )
    var test = table([35, missing, 15], ["green", "blue", "new"], color_first=True)
    var y = vector([2, -1, 5, 7, 0, -2, 12, 8, 14])
    var numeric: List[String] = ["age"]
    var categorical: List[String] = ["color"]
    var columns = ColumnTransformer(
        ColumnEntry("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric^),
        ColumnEntry("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical^),
    )
    var model = make_pipeline(columns^.as_table_transformer(), Ridge())
    comptime Model = type_of(model)
    comptime Adapter = DenseSupervisedAdapter[Model]
    comptime Table = ColumnTable[]
    var adapter = Adapter(
        apply_alpha[Model.Prefix], n_splits=3,
        input_rows=Adapter.InputRows(Table.rows, Table.take_rows),
    )
    with assert_raises(contains="non-array inputs require input_rows"):
        _ = cross_val_score[Adapter](model, train, y, Adapter())
    with assert_raises(contains="inconsistent sample counts"):
        _ = cross_val_score[Adapter](model, train, vector([0]), adapter)

    var result = ResultLine("mixed_ridge_search")
    result.add("cv", cross_val_score[Adapter](
        model, train, y, adapter, error_score=ErrorPolicy(True, 0),
    ))

    var grid = Dict[String, List[Float64]]()
    grid["ridge__alpha"] = [25.0, 1.0, 0.1]
    var search = GridSearchCV[Adapter](model, grid, adapter, error_score=ErrorPolicy(True, 0))
    _ = search.fit(train, y)
    assert_false(model.__sklearn_is_fitted__())
    assert_equal(search.n_features_in_(), 2)
    var results = search.cv_results_()
    var scores = zeros[DType.float64](Shape(3, 3))
    for candidate in range(3):
        for fold in range(3):
            scores[Item(candidate, fold)] = results.scores.test_scores.item(candidate, fold, 0)
    result.add("candidate_scores", scores)
    result.scalar("best_index", Float64(search.best_index_()))
    result.scalar("best_alpha", search.best_params_()["ridge__alpha"])
    result.scalar("best_score", search.best_score_())
    result.add("predict", search.predict(test))
    result.emit()
