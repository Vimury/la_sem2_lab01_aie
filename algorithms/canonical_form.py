# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from core.linalg import qr, matmul, transpose, svd


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [c.copy() for c in tt.cores]
    for k in range(tt.order - 1):
        core = cores[k]
        r_prev, n_k, r_curr = core.shape
        m = r_prev * n_k
        n = r_curr

        if m >= n:
            unfolded = core.reshape((m, n))
            Q, R = qr(unfolded)
            cores[k] = Q.reshape((r_prev, n_k, r_curr))

            next_core = cores[k + 1]
            r_curr_next, n_next, r_next = next_core.shape
            next_core_2d = next_core.reshape((n, n_next * r_next))
            cores[k + 1] = matmul(R, next_core_2d).reshape((n, n_next, r_next))
        else:
            unfolded = core.reshape((m, n))
            U, S, Vt = svd(unfolded, full_matrices=False)

            R_data = []
            for i in range(n):
                s_val = S.data[i]
                for j in range(n):
                    R_data.append(s_val * Vt.data[i * n + j])
            R = DenseTensor((n, n), data=R_data)

            cores[k] = U.reshape((r_prev, n_k, n))

            next_core = cores[k + 1]
            r_curr_next, n_next, r_next = next_core.shape
            next_core_2d = next_core.reshape((n, n_next * r_next))
            cores[k + 1] = matmul(R, next_core_2d).reshape((n, n_next, r_next))

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [c.copy() for c in tt.cores]
    for k in range(tt.order - 1, 0, -1):
        core = cores[k]
        r_prev, n_k, r_curr = core.shape
        n_rank = n_k * r_curr

        if r_prev <= n_rank:
            unfolded = core.reshape((r_prev, n_rank))
            unfolded_T = transpose(unfolded)
            Q_qr, R_qr = qr(unfolded_T)
            R = transpose(R_qr)
            Q = transpose(Q_qr)
            cores[k] = Q.reshape((r_prev, n_k, r_curr))

            prev_core = cores[k - 1]
            r_prev2, n_prev, r_prev1 = prev_core.shape
            prev_core_2d = prev_core.reshape((n_prev * r_prev2, r_prev1))
            cores[k - 1] = matmul(prev_core_2d, R).reshape((r_prev2, n_prev, r_prev1))
        else:
            m = r_prev
            n = n_rank
            unfolded = core.reshape((m, n))
            U, S, Vt = svd(unfolded, full_matrices=False)

            R_data = []
            for i in range(m):
                for j in range(n):
                    R_data.append(U.data[i * n + j] * S.data[j])
            R = DenseTensor((m, n), data=R_data)

            cores[k] = Vt.reshape((n, n_k, r_curr))

            prev_core = cores[k - 1]
            r_prev2, n_prev, r_prev1 = prev_core.shape
            prev_core_2d = prev_core.reshape((n_prev * r_prev2, r_prev1))
            cores[k - 1] = matmul(prev_core_2d, R).reshape((r_prev2, n_prev, n))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.size == 0:
        return 0
    max_s = S.data[0]
    threshold = max(abs_tol, rel_tol * max_s)

    rank = S.size
    for j in range(S.size):
        if S.data[j] <= threshold:
            rank = j
            break
    return rank


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
        rank:     длина диагонального вектора
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    new_data = []
    for i in range(m):
        d_val = diag_vec.data[i]
        for j in range(n):
            new_data.append(d_val * matrix.data[i * n + j])
    return DenseTensor((m, n), data=new_data)


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    m, n = matrix.shape
    new_data = []
    for i in range(m):
        for j in range(n):
            new_data.append(matrix.data[i * n + j] * diag_vec.data[j])
    return DenseTensor((m, n), data=new_data)