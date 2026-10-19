"""Bound and serialize this family's compiler processes using Linux tools."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADDRESS_SPACE_BYTES = 8 * 1024**3
COMPILER_TIMEOUT = '10m'


def compiler_command(args):
    """Wrap compilation only; runtime tests and version queries stay unchanged.

    RLIMIT_AS is a hard per-process virtual-address-space ceiling, including
    swapped pages, not an RSS measurement. Children inherit it. Allocation
    failure terminates an oversized build; timeout forcibly stops a stuck one.
    The flock is shared by root and focused univariate runners.
    """
    args = list(map(str, args))
    if not args or Path(args[0]).name != 'mojo' or len(args) < 2:
        return args
    if args[1] not in ('build', 'precompile', 'run'):
        return args
    lock = ROOT / 'builds/compiler.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    return [
        'flock', '--exclusive', '--no-fork', str(lock),
        'prlimit', f'--as={ADDRESS_SPACE_BYTES}:{ADDRESS_SPACE_BYTES}', '--core=0:0', '--',
        'timeout', '--kill-after=5s', COMPILER_TIMEOUT, *args,
    ]
