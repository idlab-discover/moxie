"""Exact target identity and class order against the pinned compiled sklearn."""
from pathlib import Path
import subprocess
import numpy as np
from numpy.testing import assert_array_equal
import sklearn
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parents[2] / 'scikit-learn'
REVISION = '4358542a1bd39af2e7e1cbace8b9fed067473c5b'
assert subprocess.check_output(['git','-C',str(REFERENCE),'rev-parse','HEAD'],text=True).strip() == REVISION
assert sklearn.__version__ == '1.10.dev0'
installed = Path(sklearn.__file__).resolve().parent
assert (installed/'neighbors/_classification.py').read_bytes() == (REFERENCE/'sklearn/neighbors/_classification.py').read_bytes()
x = np.arange(4,dtype=np.float32).reshape(-1,1)
integers = np.array([9007199254740992,9007199254740993,-9223372036854775807,9223372036854775807],dtype=np.int64)
strings = np.array([['z','red'],['a','blue'],['z','blue'],['a','red']])
raw = subprocess.check_output([str(ROOT/'builds/test_general_labels')],text=True)
lines = [line.split() for line in raw.splitlines()]
actual_int = np.array([int(row[2]) for row in lines if row[0]=='I'],dtype=np.int64)
actual_str = np.array([row[3] for row in lines if row[0]=='S']).reshape(4,2)
integer_classes = [np.array([int(row[2]) for row in lines if row[0]=='IC'],dtype=np.int64)]
string_classes = [np.array([row[3] for row in lines if row[0]=='SC' and int(row[1])==col]) for col in range(2)]
for target,actual,classes in [(integers,actual_int,integer_classes),(strings,actual_str,string_classes)]:
    model=KNeighborsClassifier(n_neighbors=1).fit(x,target)
    assert_array_equal(actual,model.predict(x))
    expected=[model.classes_] if target.ndim==1 else model.classes_
    for got,want in zip(classes,expected):
        assert_array_equal(got,want)
print('PASS pinned sklearn exact Int64 and multioutput string predictions')
