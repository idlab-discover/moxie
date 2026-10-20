#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p builds
if mojo build -I builds/deps tests/compile_fail/missing_predict.mojo -o builds/invalid-predictor > builds/missing-predict.log 2>&1; then
    echo 'ERROR: compiler accepted a Predictor without predict' >&2
    exit 1
fi
if ! grep -q "required function 'predict' is not implemented" builds/missing-predict.log; then
    cat builds/missing-predict.log >&2
    echo 'ERROR: negative compilation failed for an unexpected reason' >&2
    exit 1
fi
echo 'Missing prediction capability correctly rejected'
