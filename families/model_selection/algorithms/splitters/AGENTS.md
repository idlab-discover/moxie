# Splitters

Follow [family guidance](../../AGENTS.md). Keep seeded draws in
`moxie._random.RandomState`, labels in `moxie._labels`, and index ownership in
`moxie.model_selection._indices`. Read [scope](docs/scope.md) before changing
split ordering, label handling or RNG consumption. The focused Moxie checker
runs exact-index parity against the recorded local sklearn revision.
