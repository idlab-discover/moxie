"""ExtraTree defaults over the decision-tree owner and shared random splitter."""
from moxie.tree._classes import BaseDecisionTree

comptime ExtraTreeClassifier[label_dtype: DType = DType.float64, string_labels: Bool = False] = BaseDecisionTree[True, True, label_dtype, string_labels]
comptime ExtraTreeRegressor = BaseDecisionTree[False, True]
