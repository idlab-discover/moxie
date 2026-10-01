# Family dependency development

Both shared assembler entrypoints read current local sources. Follow the
[assembly guide](assembly.md) and the [Moxie family policy](../../../docs/family-development.md).

A family manifest records project paths, module ownership, exports, runtime
assets and licenses. It does not freeze source revisions or file hashes. Existing
`DEPENDENCIES.lock.json` manifests remain accepted as version-1 manifests; new
and migrated consumers use `DEPENDENCIES.json`. Keep `--lock` in command lines
for compatibility with existing launchers.

Remove consumer-side hash verification and Git snapshot bootstraps when adopting
this workflow. Invoke the current shared tool at its declared local path. Source
edits, including uncommitted external edits, take effect on the next build.
Missing checkouts fail explicitly. Previously cached dependencies are unused.
Keep compiler/package locks and correctness-reference revisions separately.
