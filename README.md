# moxie

Basic Mojo project using Pixi and the stable Modular channel.

## Getting started

Install [Pixi](https://pixi.sh) and a C linker such as GCC, then run:

```sh
pixi install
pixi run start
```

Build an executable with `pixi run build`; the output is `builds/moxie`.

The entry point is `src/main.mojo`. Commit `pixi.lock` to keep the
environment reproducible. Build output, benchmarks, profiling data, and
`docs/private/` are ignored by Git.
