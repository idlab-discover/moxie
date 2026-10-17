from moxie.discriminant_analysis import LinearDiscriminantAnalysis

def main() raises:
    var model = LinearDiscriminantAnalysis()
    model.partial_fit(0, 0)
