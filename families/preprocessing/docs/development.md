# Integrated preprocessing development

Preprocessing is owned and developed in the Moxie repository. Public modules
live in `../../src/moxie/preprocessing` relative to this family directory;
family engines, tests, examples, license notices and locked standalone tooling
remain here. Follow [Moxie contributor rules](../../../AGENTS.md) and the
[development policy](../../../docs/family-development.md) from the family root.
The former sibling checkout is no longer a build dependency. Preserve the
recorded scikit-learn revision, dense scope and numerical ownership contracts.

From Moxie, run `pixi run build`, `pixi run test`, and `pixi run parity`.
For focused work run `pixi run python families/preprocessing/tools/check.py test`
or replace `test` with `parity` (optionally followed by group names) or `memory`.
The family contracts and parity drivers consume the complete Moxie namespace.
