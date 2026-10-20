# Cluster ownership

Cluster development takes place in `moxie/families/cluster`. Follow
[Moxie family development](../../../docs/family-development.md) and
[contributor rules](../../../AGENTS.md). Algorithm children and the centroid
engine are internal parts of this family, with no separate repository gate.

The root Moxie manifest owns assembly. Local manifests support isolated
consumers and internal engine checks. All consume current sources. The retired
cluster checkout is historical provenance only and is never a build input.

From Moxie, run `pixi run cluster-build`, `pixi run cluster-test`, and
`pixi run cluster-parity`. The family Pixi lock is retained for reproducibility.
