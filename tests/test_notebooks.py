"""Execute the publication notebooks that need no external data.

Opt-in (they take a few minutes): HTDEBLUR_RUN_NOTEBOOKS=1 pytest tests/test_notebooks.py
Requires the `notebooks` extra (jupyter).
"""
import os

import pytest

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 'publications')

SELF_CONTAINED = [
    'fig_limiting_analysis.ipynb',
    'fig_limiting_analysis_sarah.ipynb',
    'fig_limiting_analysis_zack.ipynb',
    'fig_multiframe_schematic.ipynb',
    'fig_multiframe_operator.ipynb',
    'fig_optimization.ipynb',
    'fig_fom_2019.ipynb',
]

pytestmark = pytest.mark.skipif(not os.environ.get('HTDEBLUR_RUN_NOTEBOOKS'),
                                reason='set HTDEBLUR_RUN_NOTEBOOKS=1 to execute notebooks')


@pytest.mark.parametrize('name', SELF_CONTAINED)
def test_notebook_runs(name, tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    nbclient = pytest.importorskip('nbclient')
    monkeypatch.setenv('HTDEBLUR_FIGURES', str(tmp_path))
    nb = nbformat.read(os.path.join(NOTEBOOK_DIR, name), as_version=4)
    nbclient.NotebookClient(nb, timeout=1800, kernel_name='python3',
                            resources={'metadata': {'path': NOTEBOOK_DIR}}).execute()
