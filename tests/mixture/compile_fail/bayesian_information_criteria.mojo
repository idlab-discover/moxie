from moxie.mixture import BayesianGaussianMixture, InformationCriteria
def require_criteria[M: InformationCriteria](model: M):
    pass
def main() raises:
    var model = BayesianGaussianMixture[DType.float64]()
    require_criteria(model)
