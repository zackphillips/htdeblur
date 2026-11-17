"""Regenerate tests/fixtures/reference.npz from the pre-port (llops-backed) code.

Provenance only. Needs the private llops package and a checkout of htdeblur from
before the numpy port (commit 6a8d867 plus the fftpack fix in
blurkernel.genRandInitialization). llops at its last revision is a snake_case
refactor of the API htdeblur was written against, and its operator sums are
broken (an unsupported `cache` argument; accumulation into np.empty buffers).
llops_compat.py maps the old names and patches both bugs before htdeblur is
imported.

    PYTHONPATH=<pre-port checkout> python tests/reference/make_reference.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import llops_compat  # noqa: F401  (must precede htdeblur)
import numpy as np
import htdeblur
import htdeblur.analysis  # noqa: F401
import htdeblur.mddataset  # noqa: F401
import harness

out = harness.run_reference_cases(htdeblur)
np.savez_compressed(os.path.join(os.path.dirname(HERE), 'fixtures', 'reference.npz'), **out)
print('wrote %d arrays' % len(out))
