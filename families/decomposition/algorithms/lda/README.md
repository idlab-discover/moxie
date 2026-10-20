# LDA ownership

The implemented LDA estimator is in `../../src/moxie/decomposition/_lda.mojo`.
Its [contract](../../docs/lda.md), tests and examples belong to this family.
The former missing-Gamma diagnostic is retained under `../../provenance/algorithms/lda`;
it does not describe current support. Run `pixi run decomposition-test` and
`pixi run decomposition-parity` from Moxie, or set `MOXIE_ESTIMATORS=lda`
for focused checks.
