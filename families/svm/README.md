# SVM in Moxie

Moxie owns the pure Mojo dense SVM family under `families/svm`.
The assembled `moxie.svm` namespace exports SVC, NuSVC, SVR, NuSVR,
OneClassSVM, LinearSVC, LinearSVR and l1_min_c, with their existing typed
configuration and capability contracts.

From the Moxie root:

```sh
pixi run svm-build
pixi run svm-test
pixi run svm-parity
pixi run svm-start
```

Build compiles the complete Moxie namespace, family tests, two isolated
consumers, both engines and linear wrappers. Test includes native lifecycle,
capability rejection and assembly conflict checks. Parity compares kernel
arithmetic, both engines and public estimators with the local sklearn revision
`4358542a1bd39af2e7e1cbace8b9fed067473c5b`. Python and C++ are validation tools;
the library implementation remains pure Mojo.

- [Supported dense scope and exclusions](docs/scope.md)
- [Public API](docs/api-surface.md)
- [Kernel API](algorithms/c_svm/docs/api.md)
- [Linear API](algorithms/linear/docs/api.md)
- [Internal ownership and plumbing](docs/plumbing.md)
- [Integration and validation](docs/migration.md)
- [Source provenance](PROVENANCE.json)

The root `DEPENDENCIES.json` registers every SVM owner. Focused consumer
manifests resolve global contracts to live Moxie sources. The retired sibling
is not a build or runtime dependency. Original compatible Pixi locks, scaffold
ancestry and third-party licenses are retained. Ignored artifacts and original
Git history stay in the retired checkout.
