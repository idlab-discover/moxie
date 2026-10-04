#!/usr/bin/env bash
set -euo pipefail
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
for case in immutable escape; do
    if mojo build -I builds/deps "tests/compile_fail/fluent_${case}.mojo" -o builds/invalid-fluent > "builds/fluent-${case}.log" 2>&1; then
        echo "ERROR: compiler accepted invalid fluent borrow: ${case}" >&2
        exit 1
    fi
    if [ "$case" = immutable ]; then
        diagnostic="operand mutability 'False' doesn't match expected mutability 'True'"
    else
        diagnostic='cannot return reference with incompatible origin'
    fi
    if ! grep -q "$diagnostic" "builds/fluent-${case}.log"; then
        cat "builds/fluent-${case}.log" >&2
        exit 1
    fi
done
echo 'Fluent returns reject immutable mutation and escaping local borrows'
