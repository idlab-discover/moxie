# Public mixture API

```mojo
from moxie.mixture import (
    GaussianMixture, GaussianMixtureParams,
    BayesianGaussianMixture, BayesianGaussianMixtureParams,
    MixtureDensity, InformationCriteria,
)
```

These exports resolve to Moxie's concrete modules, not raising declarations.
Constructor defaults, methods, shapes and native adaptations are described in
the [Gaussian API](gaussian/api.md) and [Bayesian API](bayesian/api.md).
There is no public `BaseMixture`, and Bayesian does not implement
`InformationCriteria`. RNG types remain in `moxie._random`; result types remain
associated with each estimator.
