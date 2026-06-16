# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, compute_strides, multi_index_to_flat


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not cores:
            raise ValueError("Список ядер не может быть пустым")

        self.cores = cores
        self.order = len(cores)
        self.shape = tuple(core.shape[1] for core in cores)
        self.ranks = tuple([cores[0].shape[0]] + [core.shape[2] for core in cores])

        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError("Граничные ранги должны быть равны 1")

        for i in range(self.order):
            expected_shape = (self.ranks[i], self.shape[i], self.ranks[i + 1])
            if cores[i].shape != expected_shape:
                raise ValueError(f"Некорректная форма ядра {i}: ожидалось {expected_shape}, получено {cores[i].shape}")


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        import random
        if seed is not None:
            random.seed(seed)

        shape = validate_shape(shape)
        d = len(shape)

        if len(ranks) == d - 1:
            ranks = (1,) + tuple(ranks) + (1,)
        elif len(ranks) != d + 1:
            raise ValueError(f"Некорректная длина ranks: ожидалось {d + 1} или {d - 1}, получено {len(ranks)}")

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("Граничные ранги должны быть равны 1")

        cores = []
        for k in range(d):
            core_shape = (ranks[k], shape[k], ranks[k + 1])
            size = compute_size(core_shape)
            data = [random.uniform(-1, 1) for _ in range(size)]
            cores.append(DenseTensor(core_shape, data=data))

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if len(indices) != self.order:
            raise ValueError(f"Ожидался индекс длины {self.order}, получено {len(indices)}")

        res = None
        for k in range(self.order):
            ik = indices[k]
            core = self.cores[k]
            r_k = core.shape[0]
            r_k1 = core.shape[2]
            n_k = core.shape[1]

            stride_ik = r_k1
            stride_a = n_k * r_k1

            Gk_ik = [[0.0] * r_k1 for _ in range(r_k)]
            for a in range(r_k):
                for b in range(r_k1):
                    Gk_ik[a][b] = core.data[a * stride_a + ik * stride_ik + b]

            if res is None:
                res = Gk_ik
            else:
                prev_r = len(res)
                new_r = r_k1
                new_res = [[0.0] * new_r for _ in range(prev_r)]
                for i in range(prev_r):
                    for j in range(new_r):
                        s = 0.0
                        for m in range(r_k):
                            s += res[i][m] * Gk_ik[m][j]
                        new_res[i][j] = s
                res = new_res

        return res[0][0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        res_shape = self.shape
        res_size = compute_size(res_shape)
        res_data = [0.0] * res_size
        strides = compute_strides(res_shape)

        indices = [0] * self.order
        while True:
            val = self.get_element(tuple(indices))
            flat_idx = multi_index_to_flat(tuple(indices), strides)
            res_data[flat_idx] = val

            for k in range(self.order - 1, -1, -1):
                indices[k] += 1
                if indices[k] < self.shape[k]:
                    break
                indices[k] = 0
            else:
                break

        return DenseTensor(res_shape, data=res_data)

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        tt_size = self.total_storage()
        if tt_size == 0:
            return 0.0
        return full_size / tt_size

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = [
            f"TTTensor(order={self.order})",
            f"  shape: {self.shape}",
            f"  ranks: {self.ranks}",
            f"  core_sizes: {self.core_sizes()}",
            f"  total_storage: {self.total_storage()}"
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()