# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Categorical class-label bookkeeping shared by LDA and QDA.

Mirrors `sklearn.utils.multiclass.unique_labels` / `np.unique` ordering and
`sklearn.utils._encode._encode`: classes are sorted ascending and each sample is
encoded as the index of its label in `classes_`.
"""

comptime ClassLabel = Copyable & Comparable & Writable & Deinitable
"""Requirements on a class-label type: ordered, printable and copyable.

`Int` and `String` both satisfy it, so estimators accept integer or text labels.
"""


def sorted_unique[L: ClassLabel](y: List[L]) -> List[L]:
    """Return the distinct labels of `y` in ascending order (like `np.unique`)."""
    var classes = List[L]()
    for i in range(len(y)):
        # Binary search for the insertion point in the sorted prefix.
        var lo = 0
        var hi = len(classes)
        while lo < hi:
            var mid = (lo + hi) // 2
            if classes[mid] < y[i]:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(classes) and classes[lo] == y[i]:
            continue
        classes.insert(lo, y[i].copy())
    return classes^


def label_index[L: ClassLabel](classes: List[L], label: L) -> Int:
    """Index of `label` in sorted `classes`, or -1 when absent."""
    var lo = 0
    var hi = len(classes)
    while lo < hi:
        var mid = (lo + hi) // 2
        if classes[mid] < label:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(classes) and classes[lo] == label:
        return lo
    return -1


def encode_labels[L: ClassLabel](y: List[L], classes: List[L]) raises -> List[Int]:
    """Encode each label as its index in `classes`; unknown labels raise."""
    var codes = List[Int](capacity=len(y))
    for i in range(len(y)):
        var code = label_index(classes, y[i])
        if code < 0:
            raise Error(String("ValueError: y contains previously unseen label ", y[i]))
        codes.append(code)
    return codes^


def class_counts(codes: List[Int], n_classes: Int) -> List[Int]:
    """Number of samples per encoded class (like `np.bincount(minlength=K)`)."""
    var counts = List[Int](length=n_classes, fill=0)
    for i in range(len(codes)):
        counts[codes[i]] += 1
    return counts^
