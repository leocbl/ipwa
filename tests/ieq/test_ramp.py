"""Numerical tests for the ramp function."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from ipwa.ieq import ramp

def test_ramp_number() -> None:
    r, c = ramp(-0.35811366)
    assert_array_equal(r, 0)
    assert_array_equal(c, 0.35811366)

def test_ramp_array() -> None:
    r, c = ramp([-0.34238691, 0.04688151, -0.01462435])
    assert_array_equal(r, [0, 0.04688151, 0])
    assert_array_equal(c, [0.34238691, 0, 0.01462435])

def test_ramp_matrix() -> None:
    r, c = ramp([[0.31472368, 0.40579193, -0.37301318], [0.41337585, -0.40245959, -0.22150178]])
    assert_array_equal(r, [[0.31472368, 0.40579193, 0], [0.41337585, 0, 0]])
    assert_array_equal(c, [[0, 0, 0.37301318], [0, 0.40245959, 0.22150178]])

def test_ramp_tuple() -> None:
    r, c = ramp([0.31472368, 0.40579193, -0.37301318])
    assert_array_equal(r, [0.31472368, 0.40579193, 0])
    assert_array_equal(c, [0, 0, 0.37301318])

def test_ramp_empty() -> None:
    r, c = ramp([])
    assert_array_equal(r, [])
    assert_array_equal(c, [])

def test_ramp_isnumeric() -> None:
    with pytest.raises(TypeError, match="x must be a real numeric array"):
        ramp("string")

def test_ramp_isfinite() -> None:
    with pytest.raises(ValueError, match="x must contain only finite values"):
        ramp([1, np.inf, -1])

def test_ramp_isnotnan() -> None:
    with pytest.raises(ValueError, match="x must contain only finite values"):
        ramp([1, np.nan, -1])
