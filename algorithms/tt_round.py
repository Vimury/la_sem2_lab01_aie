# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize
from core.linalg import svd, matmul


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    tt_right = right_canonicalize(tt, backend)
    d = tt_right.order

    norm_G1 = tt_right.cores[0].norm()
    if d > 1:
        delta = (eps / math.sqrt(d - 1)) * norm_G1
    else:
        delta = 0.0

    cores = [c.copy() for c in tt_right.cores]

    for k in range(d - 1):
        core = cores[k]
        r_prev, n_k, r_curr = core.shape

        unfolded = core.reshape((r_prev * n_k, r_curr))
        U, S, Vt = svd(unfolded, full_matrices=False)

        r_new = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, r_new, backend)
        cores[k] = U_trunc.reshape((r_prev, n_k, r_new))

        S_trunc = _truncate_vector(S, r_new, backend)
        Vt_trunc = _truncate_rows(Vt, r_new, backend)
        remainder = _multiply_diag_matrix(S_trunc, Vt_trunc, r_new, backend)

        next_core = cores[k + 1]
        r_curr_next, n_next, r_next = next_core.shape

        next_core_2d = next_core.reshape((r_curr, n_next * r_next))
        new_next_core_2d = matmul(remainder, next_core_2d)
        cores[k + 1] = new_next_core_2d.reshape((r_new, n_next, r_next))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if delta <= 0:
        r_new = S.size
    else:
        r_new = S.size
        tail_sum = 0.0
        for j in range(S.size - 1, -1, -1):
            tail_sum += S.data[j] ** 2
            if tail_sum <= delta ** 2:
                r_new = j
            else:
                break

    if max_rank is not None:
        r_new = min(r_new, max_rank)

    return max(1, r_new)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    m, n = matrix.shape
    new_data = []
    for i in range(m):
        for j in range(rank):
            new_data.append(matrix.data[i * n + j])
    return DenseTensor((m, rank), data=new_data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    m, n = matrix.shape
    new_data = []
    for i in range(rank):
        for j in range(n):
            new_data.append(matrix.data[i * n + j])
    return DenseTensor((rank, n), data=new_data)


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    return DenseTensor((rank,), data=vector.data[:rank])


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    new_data = []
    for i in range(m):
        d_val = diag_vec.data[i]
        for j in range(n):
            new_data.append(d_val * matrix.data[i * n + j])
    return DenseTensor((m, n), data=new_data)