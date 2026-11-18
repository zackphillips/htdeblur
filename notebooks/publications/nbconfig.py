"""Shared paths for the publication notebooks.

    HTDEBLUR_FIGURES  where figures are written (default: ./figures)
    HTDEBLUR_DATA     root of the datasets / reconstruction archives (default: ~/htdeblur-data)

Data-dependent notebooks call `require(path)` before loading, which raises a
FileNotFoundError naming the missing file and the expected layout.
"""
import os

import matplotlib.pyplot as plt

FIGURE_DIR = os.path.abspath(os.path.expanduser(os.environ.get('HTDEBLUR_FIGURES', 'figures')))
DATA_DIR = os.path.abspath(os.path.expanduser(os.environ.get('HTDEBLUR_DATA', '~/htdeblur-data')))
os.makedirs(FIGURE_DIR, exist_ok=True)


def figure_path(name):
    return os.path.join(FIGURE_DIR, name)


def data_path(*parts):
    return os.path.join(DATA_DIR, *parts)


def require(path):
    """Return `path`, or raise with instructions if it does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            'Missing input: %s\nThis notebook needs the original datasets/reconstructions, which are not in the '
            'repository. Set HTDEBLUR_DATA to the directory that holds them (see notebooks/publications/README.md).' % path)
    return path


def use_style():
    """Apply the paper's 'deblur' matplotlib style if installed; otherwise keep matplotlib defaults."""
    try:
        plt.style.use('deblur')
    except OSError:
        pass


def _keep_figures_open():
    """The notebooks were written for `%matplotlib notebook`, where figures persist across cells
    (several save or restyle a figure in a later cell). Make the inline backend behave the same."""
    try:
        ip = get_ipython()  # noqa: F821 (defined inside IPython)
    except NameError:
        return
    ip.run_line_magic('config', 'InlineBackend.close_figures = False')


_keep_figures_open()
