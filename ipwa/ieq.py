"""Tools to solve the implicit equation in the ramp-based implicit representation."""

import numpy as np
from numpy.typing import ArrayLike


def ramp(x: ArrayLike) -> tuple[np.ndarray | np.float64, np.ndarray | np.float64]:
    """
    Decompose a real-valued array into complementary nonnegative parts.

    Parameters
    ----------
    x : array_like
        Finite real integer or floating-point data representable in float64.
        Scalars and arrays of any shape, including empty arrays, are supported.

    Returns
    -------
    r : numpy.ndarray or numpy.float64
        Positive part of the input, defined elementwise as r = max(x, 0).

    c : numpy.ndarray or numpy.float64
        Complementary part of the input, defined elementwise as c = max(-x, 0).

    Raises
    ------
    TypeError
        If the input is not a supported real numeric array.

    ValueError
        If the input contains NaN or infinite values.

    Notes
    -----
    For every element, the returned arrays satisfy:
        x = r - c
        r >= 0
        c >= 0
        r * c = 0

    The float64 outputs preserve the shape of the input.

    Examples
    --------
    >>> r, c = ramp([-2.0, 0.0, 3.0])
    >>> r
    array([0., 0., 3.])
    >>> c
    array([2., 0., 0.])
    """
    x = np.asarray(x)
    if x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric array")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    x = x.astype(np.float64, copy=False)
    r = np.maximum(x, 0.0)
    return r, r - x


