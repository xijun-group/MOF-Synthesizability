"""Exact 27-image kernel for the delivered molSimplify distance matrix.

The original enumerates (-1, 0, 1)^3 for the *second* atom of each
upper-triangle pair. Preserve that convention (including unwrapped inputs),
the row-vector dot products, cdist's Euclidean kernel and min reduction.
Only move invariant work out of the pair loop and batch independent cdist
calls. No nearest-image approximation or lattice/cutoff change is used.
"""
import itertools

import numpy as np
from scipy.spatial import distance


def compute_distance_matrix2(cell, cart_coords):
    n = len(cart_coords)
    matrix = np.zeros((n, n))
    if n < 2:
        return matrix

    invcell = np.linalg.inv(cell)
    supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    images = np.empty((n, 27, 3), dtype=float)
    # Keep the original scalar-vector np.dot order for bitwise parity. A
    # whole-matrix GEMM may round differently on another BLAS build.
    for j in range(1, n):
        fcoords = np.dot(cart_coords[j], invcell) + supercells
        images[j] = np.array([np.dot(frac, cell) for frac in fcoords])

    # O(n) temporary distance storage per row, rather than a 27*n*n array.
    for i in range(n - 1):
        dists = distance.cdist([cart_coords[i]], images[i + 1:].reshape(-1, 3))
        nearest = np.amin(dists.reshape(n - i - 1, 27), axis=1)
        matrix[i, i + 1:] = nearest
        matrix[i + 1:, i] = nearest
    return matrix
