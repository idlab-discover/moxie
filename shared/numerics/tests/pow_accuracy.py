"""Check Float64 powers against a rounded 100-digit decimal oracle."""

from decimal import Decimal, localcontext
import math
from pathlib import Path
import random
import struct
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def signed_bits(value):
    return struct.unpack(">q", struct.pack(">d", value))[0]


def from_bits(value):
    return struct.unpack(">d", struct.pack(">Q", value))[0]


def around(value):
    return math.nextafter(value, -math.inf), value, math.nextafter(value, math.inf)


def main():
    rng = random.Random(723619)
    tiny = math.ulp(0.0)
    normal = float.fromhex("0x1p-1022")
    largest = float.fromhex("0x1.fffffffffffffp1023")
    inf = math.inf
    nan = math.nan
    pairs = [
        (from_bits(rng.randrange(1, signed_bits(largest) + 1)), rng.uniform(-5, 5))
        for _ in range(4096)
    ]
    pairs.extend(
        (base, exponent)
        for base in (tiny, *around(normal), *around(1.0), largest)
        for exponent in (-31.4, -2, -1.1, -0.7, 0.1, 0.5, 0.7, 1.1, 2.3, 7.1, 31.4)
    )
    pairs.extend((2.0, exponent) for edge in (-1075.0, -1074.0, -1022.0, 1024.0)
                 for exponent in around(edge))
    pairs.extend((base, exponent) for base in (0.5, 2.0, 10.0)
                 for exponent in (-1075.0, -1024.0, 1024.0))
    pairs.extend((-base, exponent) for base in (tiny, normal, 0.7, 2.0, largest)
                 for exponent in (-3, -2, -1, 1, 2, 3))
    special = [
        (nan, 0.0, 1.0), (1.0, nan, 1.0), (nan, 1.0, nan), (2.0, nan, nan),
        (-2.0, 0.5, nan), (-1.0, inf, 1.0), (-1.0, -inf, 1.0),
        (-0.0, 3.0, -0.0), (-0.0, 2.0, 0.0), (-0.0, -3.0, -inf),
        (-0.0, -2.0, inf), (0.0, 0.5, 0.0), (0.0, -0.5, inf),
        (-inf, 3.0, -inf), (-inf, 2.0, inf), (-inf, -3.0, -0.0),
        (-inf, -2.0, 0.0), (inf, -0.5, 0.0), (inf, 0.5, inf),
        (0.5, inf, 0.0), (0.5, -inf, inf), (2.0, inf, inf), (2.0, -inf, 0.0),
    ]
    with localcontext() as context:
        context.prec = 100
        cases = [(base, exponent, float(
            context.power(Decimal.from_float(base), Decimal.from_float(exponent))
        )) for base, exponent in pairs]
    cases.extend(special)

    fixture = ROOT / "builds/pow_inputs.txt"
    fixture.write_text("".join(f"{signed_bits(base)} {signed_bits(exponent)}\n"
                               for base, exponent, _ in cases))
    results = subprocess.check_output([ROOT / "builds/pow_driver", fixture], text=True)
    worst = 0
    for (base, exponent, expected), result in zip(cases, results.splitlines(), strict=True):
        actual = from_bits(int(result))
        case = (base, exponent, expected, actual)
        if math.isnan(expected):
            assert math.isnan(actual), case
        elif expected == 0.0 or math.isinf(expected):
            assert signed_bits(actual) == signed_bits(expected), case
        else:
            error = abs(signed_bits(expected) - signed_bits(actual))
            assert math.isfinite(actual) and error <= 1, (*case, error)
            worst = max(worst, error)
    print(f"PASS Float64 pow, {len(cases)} pairs, max {worst} ULP")


if __name__ == "__main__":
    main()
