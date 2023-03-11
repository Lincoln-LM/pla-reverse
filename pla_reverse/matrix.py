"""GF(2) Matrix operations"""

import numpy as np
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection


def resize(mat: np.ndarray, new_shape: tuple) -> np.ndarray:
    """Copy and resize a GF(2) matrix"""
    mat_rows, mat_cols = mat.shape
    new_rows, new_cols = new_shape
    new_mat = np.zeros(new_shape, np.uint8)
    new_mat[: min(mat_rows, new_rows), : min(mat_cols, new_cols)] = mat[
        : min(mat_rows, new_rows), : min(mat_cols, new_cols)
    ]
    return new_mat


def echelon(mat: np.ndarray) -> tuple:
    """Compute reduced row echelon form of a GF(2) matrix"""
    rows, columns = mat.shape
    echelon_mat = np.copy(mat)
    transform_mat = np.identity(rows, np.uint8)
    rank = 0
    pivots = []

    for j in range(columns):
        for i in range(rank, rows):
            if echelon_mat[i, j]:
                for k in range(rows):
                    if (k != i) and echelon_mat[k, j]:
                        echelon_mat[k] ^= echelon_mat[i]
                        transform_mat[k] ^= transform_mat[i]
                echelon_mat[[i, rank]] = echelon_mat[[rank, i]]
                transform_mat[[i, rank]] = transform_mat[[rank, i]]
                pivots.append(j)
                rank += 1
                break
    return echelon_mat, transform_mat, rank, pivots


def generalized_inverse(mat: np.ndarray) -> np.ndarray:
    """Compute the generalized inverse of a GF(2) matrix"""
    _, transform_mat, rank, pivots = echelon(mat)
    transform_mat = resize(transform_mat, (mat.shape[1], mat.shape[0]))
    for i in range(rank - 1, -1, -1):
        column_index = pivots[i]
        transform_mat[[i, column_index]] = transform_mat[[column_index, i]]
    return transform_mat


def nullbasis(mat: np.ndarray) -> np.ndarray:
    """Compute the nullbasis of a GF(2) matrix"""
    mat_inv = generalized_inverse(mat)
    basis = (mat @ mat_inv) % 2
    basis = (basis + np.identity(basis.shape[0], np.uint8)) % 2
    echelon_mat, _, rank, _ = echelon(basis)
    return echelon_mat[:rank]


def nullspace(mat: np.ndarray) -> np.ndarray:
    """Compute the nullspace of a GF(2) matrix"""
    basis = nullbasis(mat)
    space = np.zeros((1 << basis.shape[0], basis.shape[1]), np.uint8)
    for k in range(1 << basis.shape[0]):
        vec = np.zeros(basis.shape[1], np.uint8)
        for i in range(basis.shape[0]):
            if (k >> i) & 1:
                vec ^= basis[i]
        space[k] = vec
    return space


def iv_matrix(shiny_rolls: int) -> np.ndarray:
    """Build (64, 60) fixed_seed -> iv_diff matrix"""
    # GF(2) mat
    mat = np.zeros((64, 60), np.uint8)
    # each bit in fixed seed
    for bit in range(64):
        seed = 1 << bit
        rng = Xoroshiro128PlusRejection(np.uint64(seed), 0)
        rng.next()  # ec
        rng.next()  # tidsid
        for _ in range(shiny_rolls):
            rng.next()  # pid
        # each iv
        for iv_i in range(6):
            # each bit in each iv
            for iv_bit in range(5):
                # [
                #   seed_0_0, seed_0_1, seed_0_2, seed_0_3, seed_0_4,
                #   seed_1_0, seed_1_1, seed_1_2, seed_1_3, seed_1_4
                # ]
                mat[bit, iv_i * 10 + iv_bit] = (int(rng.state[0]) >> iv_bit) & 1
                mat[bit, iv_i * 10 + iv_bit + 5] = (int(rng.state[1]) >> iv_bit) & 1
            rng.next()  # advance after generating iv

    return mat


def iv_const(shiny_rolls: int) -> np.array:
    """Compute the xoroshiro constant's impact on iv diffs"""
    vec = np.zeros(12 * 5, np.uint8)
    rng = Xoroshiro128PlusRejection(0)
    rng.next()  # ec
    rng.next()  # tidsid
    for _ in range(shiny_rolls):
        rng.next()  # pid
    # each iv
    for iv_i in range(6):
        # each bit in each iv
        for iv_bit in range(5):
            # [
            #   seed_0_0, seed_0_1, seed_0_2, seed_0_3, seed_0_4,
            #   seed_1_0, seed_1_1, seed_1_2, seed_1_3, seed_1_4
            # ]
            vec[iv_i * 10 + iv_bit] = (int(rng.state[0]) >> iv_bit) & 1
            vec[iv_i * 10 + iv_bit + 5] = (int(rng.state[1]) >> iv_bit) & 1
        rng.next()  # advance after generating iv

    return vec


def vec_to_int(vec):
    """Convert GF(2) vector to an integer"""
    i = 0
    for j, bit in enumerate(vec):
        i |= int(bit) << j
    return i
