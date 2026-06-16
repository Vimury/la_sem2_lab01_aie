# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from core.linalg import matmul


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    new_cores = []
    d = tt1.order

    if d == 1:
        c1, c2 = tt1.cores[0], tt2.cores[0]
        n = c1.shape[1]
        new_data = [a + b for a, b in zip(c1.data, c2.data)]
        return TTTensor([DenseTensor((1, n, 1), data=new_data)])

    for k in range(d):
        c1, c2 = tt1.cores[k], tt2.cores[k]
        r1_prev, n, r1_next = c1.shape
        r2_prev, _, r2_next = c2.shape

        if k == 0:
            R_next = r1_next + r2_next
            new_data = [0.0] * (1 * n * R_next)
            for idx, val in enumerate(c1.data):
                j, l = divmod(idx, r1_next)
                new_data[j * R_next + l] = val
            for idx, val in enumerate(c2.data):
                j, l = divmod(idx, r2_next)
                new_data[j * R_next + (l + r1_next)] = val
            new_cores.append(DenseTensor((1, n, R_next), data=new_data))

        elif k == d - 1:
            R_prev = r1_prev + r2_prev
            new_data = [0.0] * (R_prev * n * 1)
            for idx, val in enumerate(c1.data):
                i, j = divmod(idx, n)
                new_data[i * n + j] = val
            for idx, val in enumerate(c2.data):
                i, j = divmod(idx, n)
                new_data[(i + r1_prev) * n + j] = val
            new_cores.append(DenseTensor((R_prev, n, 1), data=new_data))

        else:
            R_prev = r1_prev + r2_prev
            R_next = r1_next + r2_next
            new_data = [0.0] * (R_prev * n * R_next)
            for idx, val in enumerate(c1.data):
                i, rem = divmod(idx, n * r1_next)
                j, l = divmod(rem, r1_next)
                new_data[i * n * R_next + j * R_next + l] = val
            for idx, val in enumerate(c2.data):
                i2, rem = divmod(idx, n * r2_next)
                j, l2 = divmod(rem, r2_next)
                new_data[(i2 + r1_prev) * n * R_next + j * R_next + (l2 + r1_next)] = val
            new_cores.append(DenseTensor((R_prev, n, R_next), data=new_data))

    return TTTensor(new_cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    new_cores = [c.copy() for c in tt.cores]
    new_cores[0] = new_cores[0] * float(alpha)
    return TTTensor(new_cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    new_cores = []
    d = tt1.order
    for k in range(d):
        c1, c2 = tt1.cores[k], tt2.cores[k]
        r1_prev, n, r1_next = c1.shape
        r2_prev, _, r2_next = c2.shape
        new_R1 = r1_prev * r2_prev
        new_R2 = r1_next * r2_next
        new_data = [0.0] * (new_R1 * n * new_R2)

        for i in range(n):
            for r1 in range(r1_prev):
                for c1_idx in range(r1_next):
                    val1 = c1.data[r1 * n * r1_next + i * r1_next + c1_idx]
                    for r2 in range(r2_prev):
                        for c2_idx in range(r2_next):
                            val2 = c2.data[r2 * n * r2_next + i * r2_next + c2_idx]

                            new_r = r1 * r2_prev + r2
                            new_c = c1_idx * r2_next + c2_idx
                            new_data[new_r * n * new_R2 + i * new_R2 + new_c] = val1 * val2

        new_cores.append(DenseTensor((new_R1, n, new_R2), data=new_data))

    return TTTensor(new_cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = tt1.order
    Z = None
    for k in range(d):
        c1, c2 = tt1.cores[k], tt2.cores[k]
        r1_prev, n, r1_next = c1.shape
        r2_prev, _, r2_next = c2.shape

        if Z is None:
            Z = DenseTensor((1, 1), data=[1.0])

        new_Z_data = [0.0] * (r1_next * r2_next)
        for i in range(n):
            # c1_i^T: (r1_next, r1_prev)
            c1_i_T_data = []
            for r in range(r1_next):
                for c in range(r1_prev):
                    c1_i_T_data.append(c1.data[c * n * r1_next + i * r1_next + r])
            c1_i_T = DenseTensor((r1_next, r1_prev), data=c1_i_T_data)

            # c2_i: (r2_prev, r2_next)
            c2_i_data = []
            for r in range(r2_prev):
                for c in range(r2_next):
                    c2_i_data.append(c2.data[r * n * r2_next + i * r2_next + c])
            c2_i = DenseTensor((r2_prev, r2_next), data=c2_i_data)

            temp = matmul(c1_i_T, Z)
            res = matmul(temp, c2_i)

            for idx in range(len(res.data)):
                new_Z_data[idx] += res.data[idx]

        Z = DenseTensor((r1_next, r2_next), data=new_Z_data)

    return Z.data[0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    return math.sqrt(tt_dot(tt, tt, backend))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    neg_tt2 = tt_scalar_mul(tt2, -1.0, backend)
    diff = tt_add(tt1, neg_tt2, backend)
    return tt_norm(diff, backend)