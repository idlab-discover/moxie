"""Compare the live Numax scalar kernels with high-precision decimal results."""

from decimal import Decimal, localcontext
import math
from pathlib import Path
import random
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SEED = 723619


def bits(value):
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def float_from_bits(value):
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def around(value):
    return (math.nextafter(value, -math.inf), value, math.nextafter(value, math.inf))


def check(operation, inputs, reference, ulps):
    fixture = ROOT / "builds/libm_inputs.txt"
    fixture.write_text("".join(
        f"{operation} {struct.unpack('>q', struct.pack('>d', x))[0]}\n"
        for x in inputs
    ))
    output = subprocess.check_output([ROOT / "builds/libm_driver", fixture], text=True)
    results = output.splitlines()
    assert len(results) == len(inputs)
    worst = 0
    for x, raw in zip(inputs, results, strict=True):
        actual_bits = int(raw)
        expected = reference(x)
        if math.isnan(expected):
            assert math.isnan(float_from_bits(actual_bits)), (operation, x, raw)
        elif expected == 0 or math.isinf(expected) or (operation == "exp" and x == 0):
            assert actual_bits == bits(expected), (operation, x.hex(), expected, raw)
        else:
            actual = float_from_bits(actual_bits)
            error = abs(actual_bits - bits(expected))
            assert math.isfinite(actual) and error <= ulps, (
                operation, x.hex(), expected.hex(), actual.hex(), error
            )
            worst = max(worst, error)
    print(f"PASS Float64 {operation}, {len(inputs)} values, max {worst} ULP")


def main():
    rng = random.Random(SEED)
    special = [-math.inf, -1.0, -0.0, 0.0, 1.0, math.inf, math.nan]
    with localcontext() as context:
        context.prec = 100
        ln2 = Decimal(2).ln()
        # Range reduction, normal/subnormal output, rounding to zero and overflow.
        exp_inputs = special + [rng.uniform(-746, 710) for _ in range(4096)]
        for k in range(-1075, 1024):
            exp_inputs.extend(around(float((Decimal(k) + Decimal("0.5")) * ln2)))
        exp_inputs.extend(around(float(-1075 * ln2)))
        exp_inputs.extend(around(float(-1022 * ln2)))
        exp_inputs.extend(around(float((Decimal(2)**1024 - Decimal(2)**970).ln())))

        min_normal = math.ldexp(1.0, -1022)
        max_finite = float.fromhex("0x1.fffffffffffffp1023")
        log_inputs = special + list(around(1.0)) + list(around(min_normal))
        log_inputs.extend([math.ulp(0.0), max_finite])
        log_inputs.extend(float_from_bits(rng.randrange(1, bits(max_finite) + 1))
                          for _ in range(4096))
        for exponent in range(-1022, 1024):
            log_inputs.extend(around(math.ldexp(math.sqrt(2.0), exponent)))

        def exp_reference(value):
            return float(Decimal.from_float(value).exp())

        def log_reference(value):
            if value < 0:
                return math.nan
            return float(Decimal.from_float(value).ln())

        check("exp", exp_inputs, exp_reference, ulps=1)
        check("log", log_inputs, log_reference, ulps=1)


if __name__ == "__main__":
    main()
