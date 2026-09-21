# Shared random engine

This project owns moxie._random. Reuse the pinned foundation.assembly launcher
at ../assembly/tools/consume.py; do not introduce another assembler.
Use the numerics owner's NuMojo NDArray, Shape and Item for dense results.
RNG state and index bookkeeping are algorithmic storage, not a generic array.
Follow docs/random.md and docs/scope.md. Library code is pure Mojo.
Run pixi run build, test and parity; never update dependency hashes during builds.
