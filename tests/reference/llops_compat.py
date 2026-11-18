"""Make llops (last revision) run the pre-port htdeblur code. Reference generation only.

1. Map the 2019 names htdeblur calls onto the snake_case refactor
   (Dstack -> DiagStack, VecStack -> vector_stack, TV -> TotalVariation, ...).
2. Patch llops bugs: removed numpy/collections APIs, operator sums passing an
   unsupported `cache` kwarg, and alloc() returning np.empty buffers that sums
   accumulate into.
"""
import sys, llops, llops.operators as ops, llops.operators.solvers as s, llops.filter as filt, llops.display as disp, llops.roi as roi
sys.modules['llops.solvers'] = s
for m in ('iterative', 'objectivefunctions', 'regularizers'):
    sys.modules['llops.solvers.' + m] = getattr(s, m)
llops.expandDims = llops.expand_dims
llops.sanatizeDictionary = llops.util.sanatize_dictionary
filt.downsample = llops.downsample
filt.upsample = llops.upsample
disp.listPlotFlat = disp.list_plot_flat
roi.getOverlapRegion = roi.get_overlapping_region
ops.Dstack = ops.DiagStack
ops.Vstack = ops.VStack
ops.VecStack = ops.vector_stack
ops.VecSplit = ops.vector_split
ops._GradientOperator = ops.operators.GradientOperator
s.regularizers.TV = s.regularizers.TotalVariation
import numpy as _np; _np.asscalar = lambda a: _np.asarray(a).item()
import collections, collections.abc; collections.Iterable = collections.abc.Iterable
# 2019 API: `op.inv * x`; HEAD: `op.inv(x)` / `op.inverse_op()`
from llops.operators.operators import Operator as _Op
class _Inv:
    def __init__(self, op): self.op = op
    def __mul__(self, x): return self.op.inverse(x)
_Op.inv = property(lambda self: _Inv(self))
import llops.operators.operators as _oo; _oo.cache = None
# llops HEAD: operator sums call A._forward(x, y, cache=...), which no operator accepts. Drop the kwarg.
_orig_init = _Op.__init__
def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    f = self._forward
    if f is not None and not getattr(f, '_cache_ok', False):
        def _fwd(x, y, cache=None, _f=f):
            return _f(x, y)
        _fwd._cache_ok = True
        self._forward = _fwd
_Op.__init__ = _patched_init
# llops HEAD: alloc() returns np.empty, and operator sums accumulate into it with +=. Zero it.
import llops.base as _b
def _alloc(shape, dtype, backend):
    return _np.zeros(shape, dtype=_b.getNativeDatatype(dtype, backend), order='F')
_b.alloc = _alloc
for _m in list(sys.modules.values()):
    if _m is not None and getattr(_m, '__name__', '').startswith('llops') and hasattr(_m, 'alloc'):
        _m.alloc = _alloc
# 2019 API: Dstack[i] and Dstack.stack_operators
ops.DiagStack.stack_operators = property(lambda self: [row[i] for i, row in enumerate(self.operators)])
ops.DiagStack.__getitem__ = lambda self, i: self.stack_operators[i]
