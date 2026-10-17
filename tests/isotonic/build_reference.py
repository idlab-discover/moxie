"""Compile the exact pinned local Cython engine in consumer-owned storage."""
from pathlib import Path
from setuptools import Extension, setup
from Cython.Build import cythonize
ROOT = Path(__file__).resolve().parents[2]
source = ROOT.parent / 'scikit-learn/sklearn/_isotonic.pyx'
setup(name='moxie-isotonic-oracle', ext_modules=cythonize(
    [Extension('_moxie_isotonic_reference', [str(source)])],
    build_dir=str(ROOT/'builds/isotonic/reference/cython'), compiler_directives={'language_level': 3}))
