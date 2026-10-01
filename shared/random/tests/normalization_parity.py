"""Verify native normalization against the original sklearn normalizer."""
from pathlib import Path
import json
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(check):
    modes = ('integer', 'optional_integer', 'borrowed', 'default', 'none', 'optional_none')
    seeds = (0, 1, 42, 5489, 2**32 - 1)
    saved = np.random.get_state()
    cases = 0
    try:
        for seed in seeds:
            for mode in modes:
                out = subprocess.check_output([ROOT/'builds/normalization_draws',mode,str(seed)],text=True)
                lines = iter(out.splitlines())
                def state_matches(state):
                    assert next(lines) == 'STATE'
                    reference = state.get_state()
                    assert int(next(lines)) == reference[2]
                    assert int(next(lines)) == reference[3]
                    np.testing.assert_allclose(float(next(lines)),reference[4],rtol=2e-14,atol=2e-14)
                    words = np.array([int(next(lines)) for _ in range(624)],dtype=np.uint32)
                    np.testing.assert_array_equal(words,reference[1])
                borrowed = np.random.RandomState(seed)
                borrowed.standard_normal()
                np.random.seed(seed)
                check(None).standard_normal()
                config = seed if mode in ('integer','optional_integer') else borrowed if mode == 'borrowed' else None
                for _ in range(3):
                    rng = check(config)
                    assert next(lines) == 'DRAWS'
                    uniforms = np.array([float(next(lines)) for _ in range(17)])
                    normals = np.array([float(next(lines)) for _ in range(19)])
                    integers = np.array([int(next(lines)) for _ in range(5)])
                    np.testing.assert_array_equal(uniforms,rng.random_sample(17))
                    np.testing.assert_allclose(normals,rng.standard_normal(19),rtol=2e-14,atol=2e-14)
                    np.testing.assert_array_equal(integers,rng.randint(1000,size=5))
                    state_matches(rng)
                assert next(lines) == 'BORROWED'
                state_matches(borrowed)
                assert next(lines) == 'GLOBAL'
                state_matches(check(None))
                assert next(lines,None) is None
                cases += 1
    finally:
        np.random.set_state(saved)
    (ROOT/'builds/normalization-parity.json').write_text(json.dumps(
        dict(cases=cases,rounds_per_case=3,seeds=seeds,modes=modes,numpy=np.__version__,
             oracle='pinned sklearn.utils.validation.check_random_state',
             compared='all outputs, 624 MT words, position, Gaussian cache and unaffected streams'),indent=2)+'\n')
    print(f'PASS {cases} normalization cases x 3 rounds against pinned sklearn, including complete state')
