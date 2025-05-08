import numpy as np


def zigzag_scan(block):
    if not isinstance(block, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if block.ndim != 2:
        raise ValueError("Input must be 2D")
    h, w = block.shape
    if h != w:
        raise ValueError("Input must be a square matrix")

    n = h
    result = np.empty(n * n, dtype=block.dtype)  # Предварительное выделение памяти
    index = 0
    row, col = 0, 0
    going_up = True

    for _ in range(n * n):
        result[index] = block[row, col]  # Заполняем результат
        index += 1

        # Переход по диагоналям с изменением направления
        if going_up:
            if col == n - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == n - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1

    return result


def inverse_zigzag_scan(arr, n):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if arr.ndim != 1:
        raise ValueError("Input must be 1D array")
    if arr.size != n * n:
        raise ValueError("Input size must be n*n")
    if n <= 0:
        raise ValueError("n must be positive")

    block = np.zeros((n, n), dtype=arr.dtype)
    idx = 0
    row, col = 0, 0
    going_up = True

    for _ in range(n * n):
        block[row, col] = arr[idx]
        idx += 1

        # Переход по диагоналям с изменением направления
        if going_up:
            if col == n - 1:
                row += 1
                going_up = False
            elif row == 0:
                col += 1
                going_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == n - 1:
                col += 1
                going_up = True
            elif col == 0:
                row += 1
                going_up = True
            else:
                row += 1
                col -= 1

    return block
