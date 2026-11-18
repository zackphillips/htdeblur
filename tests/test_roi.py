from htdeblur.roi import Roi, get_overlapping_region
import numpy as np


def test_union_and_intersection():
    a, b = Roi(start=(0, 0), shape=(10, 10)), Roi(start=(5, 6), shape=(10, 10))
    assert (a + b).start == (0, 0) and (a + b).end == (15, 16)
    assert (a & b).start == (5, 6) and (a & b).end == (10, 10)


def test_offset_and_compose():
    outer = Roi(start=(2, 3), shape=(10, 12), input_shape=(40, 40))
    inner = Roi(start=(1, 1), shape=(4, 4), input_shape=(10, 12))
    composed = outer * inner
    assert composed.start == (3, 4) and composed.shape == (4, 4) and composed.input_shape == (40, 40)
    assert (outer + (1, -1)).start == (3, 2)
    assert (outer - (2, 3)).start == (0, 0)


def test_slice_clips_to_input_shape():
    r = Roi(start=(-2, 5), shape=(6, 10), input_shape=(8, 12))
    assert r.slice == (slice(0, 4), slice(5, 12))
    assert r.valid.shape == (4, 7)


def test_overlap_region():
    img = np.arange(100).reshape(10, 10)
    a, b = Roi(start=(0, 0), shape=(10, 10)), Roi(start=(0, 6), shape=(10, 10))
    o0, o1 = get_overlapping_region((img, img), (a, b))
    assert o0.shape == o1.shape == (10, 4)
    np.testing.assert_array_equal(o0, img[:, 6:10])
