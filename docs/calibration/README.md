# Calibration

Moxie owns `CalibratedClassifierCV`, `calibration_curve`, and
`CalibrationDisplay`, including sigmoid, isotonic and temperature calibration,
materialized CV splits and native SVG displays.

See the [API](api-surface.md), [contracts and scope](family.md), and
[runnable example](../../examples/calibration.mojo).

```sh
pixi run python tools/calibration_checks.py test
pixi run python tools/calibration_checks.py parity
```

The same checks run through root `build`, `test` and `parity` commands. The
parity checker uses the recorded scikit-learn source revision and exact
validation wheels. It requires `uv` to create its isolated oracle environment.
The retired calibration sibling is not required.

The side-by-side notebook
[`notebooks/calibration/plot_calibration.ipynb`](../../notebooks/calibration/plot_calibration.ipynb)
replays `examples/calibration/plot_calibration.py` with Moxie's `GaussianNB`
behind a `CalibrationEstimator` adapter. Rerun it with
`pixi run -e notebooks notebooks notebooks/calibration`.
