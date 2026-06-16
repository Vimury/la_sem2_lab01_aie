# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from core.linalg import svd


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = tensor.ndim
    if d == 1:
        return TTTensor([tensor.reshape((1, tensor.shape[0], 1))])

    norm_A = tensor.norm()
    if norm_A > 1e-30:
        delta = (eps / math.sqrt(d - 1)) * norm_A
    else:
        delta = 0.0

    C = tensor.copy()
    cores = []
    r_prev = 1

    for k in range(d - 1):
        if k == 0:
            n_k = tensor.shape[0]
            C_unfolded = C.unfolding(0)
        else:
            n_k = tensor.shape[k]
            rest_size = C.shape[1]
            C_unfolded = C.reshape((r_prev * n_k, rest_size // n_k))

        U, S, Vt = svd(C_unfolded, full_matrices=False)

        r_k = _compute_truncated_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, r_k, backend)
        cores.append(U_trunc.reshape((r_prev, n_k, r_k)))

        S_trunc = _truncate_vector(S, r_k, backend)
        Vt_trunc = _truncate_rows(Vt, r_k, backend)

        C = _multiply_diag_matrix(S_trunc, Vt_trunc, r_k, backend)
        r_prev = r_k

    n_last = tensor.shape[-1]
    cores.append(C.reshape((r_prev, n_last, 1)))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    tol = max(1e-12, 1e-8 * S.data[0]) if S.size > 0 else 1e-12
    num_rank = S.size
    for j in range(S.size):
        if S.data[j] <= tol:
            num_rank = j
            break

    r_k = num_rank
    if delta > 0:
        tail_sum = 0.0
        for j in range(num_rank - 1, -1, -1):
            tail_sum += S.data[j] ** 2
            if tail_sum <= delta ** 2:
                r_k = j
            else:
                break

    if max_rank is not None:
        r_k = min(r_k, max_rank)

    return max(1, r_k)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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