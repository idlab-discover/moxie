"""Source changes must reach the next build without repinning or a cache fallback."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1] / 'shared/assembly/tools'))
from family_validator import assemble


def main():
    with tempfile.TemporaryDirectory(dir=ROOT / 'builds', prefix='live-sources-') as folder:
        consumer = Path(folder)
        source = consumer / 'probe.mojo'
        manifest = consumer / 'DEPENDENCIES.json'
        manifest.write_text(json.dumps({'schema': 1, 'overrides': [], 'projects': [{
            'id': 'probe', 'path': '.', 'assets': {},
            'modules': {'moxie.probe': {'file': 'probe.mojo'}},
            'exports': {'schema': 1, 'packages': {'moxie': {}}},
        }]}))
        output = consumer / 'builds/deps'
        for content in ['# first source\n', '# edited source without repinning\n']:
            source.write_text(content)
            assemble(consumer, manifest, output)
            assert (output / 'moxie/probe.mojo').read_text() == content
    print('PASS: live source edits consumed on the next build')


if __name__ == '__main__':
    main()
