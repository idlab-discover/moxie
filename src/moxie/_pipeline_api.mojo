# Adapted from scikit-learn; authors: scikit-learn developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Public scikit-learn API declarations; all new operations are deferred."""
from moxie._pipeline_api_types import APIValue

def make_pipeline(*steps: Some[Movable], memory: Some[Movable] = None, transform_input: Some[Movable] = ('X_val',), verbose: Some[Movable] = False) raises -> APIValue:
    "Scikit-learn signature: make_pipeline(*steps, memory=None, transform_input=('X_val',), verbose=False)."
    raise Error("NotImplementedError: make_pipeline is an API scaffold")

def make_union(*transformers: Some[Movable], n_jobs: Some[Movable] = None, verbose: Some[Movable] = False, verbose_feature_names_out: Some[Movable] = True) raises -> APIValue:
    "Scikit-learn signature: make_union(*transformers, n_jobs=None, verbose=False, verbose_feature_names_out=True)."
    raise Error("NotImplementedError: make_union is an API scaffold")
