# Decomposition engine ownership

Follow [family rules](../../AGENTS.md). This directory owns
`moxie.decomposition._svd_engine` and the `randomized_svd` export.
Use family-owned contracts and Moxie's shared storage and RNG owners.
Run `python shared/svd_engine/tools/check.py test` and `parity` from the
family directory in Moxie's Pixi environment. Do not refresh source pins.
