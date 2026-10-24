"""Shared estimator kinds: trait conformance mirrors sklearn.base.is_* predicates.

sklearn's is_classifier/is_regressor/is_clusterer/is_outlier_detector read the
estimator_type tag. Moxie expresses that type through the shared mixin traits, so
each concrete estimator must conform to exactly the mixins whose sklearn
predicate is true (checked against the pinned reference classes). Pipeline and
SelfTrainingClassifier are classifiers/regressors in sklearn but do not declare
the mixins yet; see the shared_surface tracking pull request.
"""
from std.testing import assert_equal
from moxie.base import ClassifierMixin, RegressorMixin, ClusterMixin, OutlierMixin
from moxie.dummy import DummyClassifier, DummyRegressor
from moxie.naive_bayes import GaussianNB, MultinomialNB
from moxie.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR, OneClassSVM
from moxie.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, NearestCentroid, LocalOutlierFactor, KernelDensity
from moxie.linear_model import LinearRegression, Ridge, RidgeClassifier, Lasso, ElasticNet, BayesianRidge
from moxie.tree import DecisionTreeClassifier, DecisionTreeRegressor
from moxie.ensemble import RandomForestClassifier, RandomForestRegressor
from moxie.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from moxie.cluster import KMeans, DBSCAN, Birch, MeanShift, AgglomerativeClustering
from moxie.covariance import EllipticEnvelope
from moxie.kernel_ridge import KernelRidge
from moxie.isotonic import IsotonicRegression
from moxie.cross_decomposition import PLSRegression
from moxie.gaussian_process import GaussianProcessRegressor
from moxie.mixture import GaussianMixture
from moxie.semi_supervised import LabelPropagation

def kinds[E: AnyType]() -> String:
    """Space-separated sklearn predicates that hold for E, in sklearn.base order."""
    var s = String()
    comptime if conforms_to(E, ClassifierMixin):
        s += " classifier"
    comptime if conforms_to(E, RegressorMixin):
        s += " regressor"
    comptime if conforms_to(E, ClusterMixin):
        s += " clusterer"
    comptime if conforms_to(E, OutlierMixin):
        s += " outlier_detector"
    return s


def main() raises:
    assert_equal(kinds[DummyClassifier[]](), " classifier", "DummyClassifier")
    assert_equal(kinds[DummyRegressor[]](), " regressor", "DummyRegressor")
    assert_equal(kinds[GaussianNB[]](), " classifier", "GaussianNB")
    assert_equal(kinds[MultinomialNB[]](), " classifier", "MultinomialNB")
    assert_equal(kinds[SVC[]](), " classifier", "SVC")
    assert_equal(kinds[NuSVC[]](), " classifier", "NuSVC")
    assert_equal(kinds[LinearSVC[]](), " classifier", "LinearSVC")
    assert_equal(kinds[SVR](), " regressor", "SVR")
    assert_equal(kinds[NuSVR](), " regressor", "NuSVR")
    assert_equal(kinds[LinearSVR](), " regressor", "LinearSVR")
    assert_equal(kinds[OneClassSVM](), " outlier_detector", "OneClassSVM")
    assert_equal(kinds[KNeighborsClassifier[]](), " classifier", "KNeighborsClassifier")
    assert_equal(kinds[KNeighborsRegressor[]](), " regressor", "KNeighborsRegressor")
    assert_equal(kinds[RadiusNeighborsClassifier[]](), " classifier", "RadiusNeighborsClassifier")
    assert_equal(kinds[RadiusNeighborsRegressor[]](), " regressor", "RadiusNeighborsRegressor")
    assert_equal(kinds[NearestCentroid[]](), " classifier", "NearestCentroid")
    assert_equal(kinds[LocalOutlierFactor[]](), " outlier_detector", "LocalOutlierFactor")
    assert_equal(kinds[KernelDensity[]](), "", "KernelDensity")
    assert_equal(kinds[LinearRegression[]](), " regressor", "LinearRegression")
    assert_equal(kinds[Ridge[]](), " regressor", "Ridge")
    assert_equal(kinds[RidgeClassifier[]](), " classifier", "RidgeClassifier")
    assert_equal(kinds[Lasso[]](), " regressor", "Lasso")
    assert_equal(kinds[ElasticNet[]](), " regressor", "ElasticNet")
    assert_equal(kinds[BayesianRidge[]](), " regressor", "BayesianRidge")
    assert_equal(kinds[DecisionTreeClassifier[]](), " classifier", "DecisionTreeClassifier")
    assert_equal(kinds[DecisionTreeRegressor](), " regressor", "DecisionTreeRegressor")
    assert_equal(kinds[RandomForestClassifier[]](), " classifier", "RandomForestClassifier")
    assert_equal(kinds[RandomForestRegressor](), " regressor", "RandomForestRegressor")
    assert_equal(kinds[LinearDiscriminantAnalysis[]](), " classifier", "LinearDiscriminantAnalysis")
    assert_equal(kinds[QuadraticDiscriminantAnalysis[]](), " classifier", "QuadraticDiscriminantAnalysis")
    assert_equal(kinds[KMeans[]](), " clusterer", "KMeans")
    assert_equal(kinds[DBSCAN[]](), " clusterer", "DBSCAN")
    assert_equal(kinds[Birch[]](), " clusterer", "Birch")
    assert_equal(kinds[MeanShift[]](), " clusterer", "MeanShift")
    assert_equal(kinds[AgglomerativeClustering[]](), " clusterer", "AgglomerativeClustering")
    assert_equal(kinds[EllipticEnvelope[]](), " outlier_detector", "EllipticEnvelope")
    assert_equal(kinds[KernelRidge[]](), " regressor", "KernelRidge")
    assert_equal(kinds[IsotonicRegression[]](), " regressor", "IsotonicRegression")
    assert_equal(kinds[PLSRegression[]](), " regressor", "PLSRegression")
    assert_equal(kinds[GaussianProcessRegressor](), " regressor", "GaussianProcessRegressor")
    assert_equal(kinds[GaussianMixture[]](), "", "GaussianMixture")
    assert_equal(kinds[LabelPropagation[]](), " classifier", "LabelPropagation")
    print("Estimator kinds match sklearn.base predicates for", 42, "estimators")
