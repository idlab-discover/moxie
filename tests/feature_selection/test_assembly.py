"""The launcher must run the live shared assembler named by the manifest, without pins."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('assembly', Path(__file__).resolve().parents[2] / 'tools/feature_selection_assembly.py')
assembly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembly)


class LiveLauncherTests(unittest.TestCase):
    def test_live_tool_and_wrong_assembly_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'DEPENDENCIES.json'
            calls = []
            def run(*args, **kwargs):
                calls.append(args)
            path.write_text(json.dumps(dict(schema=1, overrides=[], projects=[
                dict(id='assembly', path=str(assembly.ASSEMBLY), modules={}, assets={},
                     exports={'schema': 1, 'packages': {}})])))
            assembly.assemble(root, run)
            self.assertEqual(len(calls), 1)
            self.assertEqual(Path(calls[0][1]), assembly.ASSEMBLY / 'tools/consume_family.py')
            self.assertEqual(Path(calls[0][calls[0].index('--lock') + 1]), path.resolve())
            self.assertEqual(Path(calls[0][calls[0].index('--tool') + 1]), assembly.ASSEMBLY / 'tools/assemble_family.py')
            self.assertTrue((assembly.ASSEMBLY / 'tools/consume_family.py').is_file())
            path.write_text(json.dumps(dict(schema=1, overrides=[], projects=[
                dict(id='assembly', path=str(root / 'cached'), modules={}, assets={},
                     exports={'schema': 1, 'packages': {}})])))
            with self.assertRaisesRegex(ValueError, 'must point at'):
                assembly.assemble(root, run)
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
