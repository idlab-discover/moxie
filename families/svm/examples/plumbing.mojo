"""Own a training snapshot without fitting an SVM."""
import numojo as nm
from moxie.svm._contracts import TrainingSnapshot, pair_index

def main() raises:
    var x = nm.ones[DType.float64](nm.Shape(4, 2))
    var snapshot = TrainingSnapshot(x)
    snapshot.validate_query(x)
    print("training rows:", snapshot.n_train)
    print("pair (1, 2) in three classes:", pair_index(3, 1, 2))
