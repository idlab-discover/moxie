from moxie.impute._plumbing import missing_mask, MissingValue

def main() raises:
    _ = missing_mask([1, 2], MissingValue[DType.float64]())
