"""Exercise kernel memory limits and compiler serialization without large loads."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import compiler_limits as limits


class CompilerLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(limits, 'ROOT', self.root)
        self.root_patch.start()
        self.compiler = self.root / 'mojo'
        self.compiler.write_text(f'#!{sys.executable}\n' + '''import mmap, resource, signal, sys, time
mode = sys.argv[2]
if mode == 'memory':
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert soft == hard == 8 * 1024**3
    assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
    try:
        mmap.mmap(-1, hard + 4096)
    except (OSError, MemoryError):
        print('allocation blocked')
    else:
        raise AssertionError('allocation exceeded the cap')
elif mode == 'lock':
    with open(sys.argv[3], 'a') as f:
        f.write('start ' + sys.argv[4] + '\\n'); f.flush()
        time.sleep(0.2)
        f.write('end ' + sys.argv[4] + '\\n')
elif mode == 'timeout':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
elif mode == 'fail':
    sys.exit(23)
''')
        self.compiler.chmod(0o755)

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def command(self, *args):
        return limits.compiler_command([self.compiler, 'build', *args])

    def test_actual_address_space_limit(self):
        result = subprocess.run(self.command('memory'), capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout.strip(), 'allocation blocked')

    def test_compilers_do_not_overlap(self):
        events = self.root / 'events'
        first = subprocess.Popen(self.command('lock', events, 'a'))
        second = subprocess.Popen(self.command('lock', events, 'b'))
        try:
            self.assertEqual(first.wait(timeout=10), 0)
            self.assertEqual(second.wait(timeout=10), 0)
        finally:
            for proc in (first, second):
                if proc.poll() is None:
                    proc.kill()
                proc.wait()
        self.assertIn(events.read_text().splitlines(), [
            ['start a', 'end a', 'start b', 'end b'],
            ['start b', 'end b', 'start a', 'end a'],
        ])

    def test_timeout_kills_and_releases_lock(self):
        with patch.object(limits, 'COMPILER_TIMEOUT', '0.2s'):
            result = subprocess.run(self.command('timeout'), timeout=10)
        self.assertIn(result.returncode, (-9, 137))
        result = subprocess.run(['flock', '--nonblock', str(self.root / 'builds/compiler.lock'), 'true'], timeout=2)
        self.assertEqual(result.returncode, 0)

    def test_exit_status_and_noncompiler_commands(self):
        result = subprocess.run(self.command('fail'), timeout=10)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(limits.compiler_command(['mojo', '--version']), ['mojo', '--version'])
        self.assertEqual(limits.compiler_command(['python', 'test.py']), ['python', 'test.py'])
        self.assertEqual(limits.compiler_command([]), [])


if __name__ == '__main__':
    unittest.main()
