# Assembly owner

Before changing lock semantics, read docs/assembly.md and schemas/assembly-v1.schema.json.
Keep the tool in tools/assemble.py; consumers execute its pinned path read-only.
Reuse parent moxie.base and moxie.utils.validation, and numerics moxie._dense
(sum_axis2d, row_squared_norms), numojo.NDArray, Shape and Item. Numerical
policy is ../numerics/docs/numerics.md. This project owns no numerical kernels.
Run pixi run build, test and parity after changes. Fixtures stay in tests/fixtures;
all generated namespaces stay under the respective consumer's builds/deps.
