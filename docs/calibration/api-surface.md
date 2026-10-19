# Public API

`moxie.calibration` exports implemented `CalibratedClassifierCV`,
`calibration_curve`, and `CalibrationDisplay`.

Native estimator and split types are `CalibratedClassifier`,
`CalibrationEstimator`, `CalibratedClassifierParams`, `CalibrationCV` and
`CalibrationSplit`. `FrozenCalibrationEstimator` bridges Moxie frozen handles. Native display types are `CalibrationDisplayEstimator`,
`CalibrationAxes`, `CalibrationFigure` and `CalibrationLine`.

See [the family guide](family.md) for supported inputs, examples and explicit
differences from Python scikit-learn. No public operation uses an opaque
placeholder result type.
