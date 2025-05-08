import numpy as np

def _c_factor(k):
    return 1.0 / np.sqrt(2) if k == 0 else 1.0

def _build_dct_matrix(size):
    n = np.arange(size)
    k = n.reshape(-1, 1)
    return np.cos((2 * n + 1) * k * np.pi / (2 * size))

def dct2(block):
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError("Блок должен быть квадратным.")

    N = block.shape[0]
    block = block.astype(np.float64)
    if block.dtype == np.uint8:
        block -= 128

    T = _build_dct_matrix(N)
    dct_core = T @ block @ T.T

    C = np.array([_c_factor(i) for i in range(N)])
    scale = np.outer(C, C)

    return 0.25 * scale * dct_core

def idct2(coeffs):
    if coeffs.ndim != 2 or coeffs.shape[0] != coeffs.shape[1]:
        raise ValueError("Коэффициенты должны быть в квадратном блоке.")

    N = coeffs.shape[0]
    T = _build_dct_matrix(N)

    C = np.array([_c_factor(i) for i in range(N)])
    scale = np.outer(C, C)

    restored = T.T @ (coeffs * scale) @ T
    return 0.25 * restored

def dct2_matrix(block):
    N = block.shape[0]
    if block.shape[1] != N:
        raise ValueError("Блок должен быть квадратным.")

    block = block.astype(np.float64)
    if block.dtype == np.uint8:
        block -= 128

    T = _build_dct_matrix(N)
    dct_unscaled = T @ block @ T.T

    C = np.array([_c_factor(i) for i in range(N)])
    scale = np.outer(C, C)

    return 0.25 * scale * dct_unscaled

def idct2_matrix(dct_coeffs):
    N = dct_coeffs.shape[0]
    if dct_coeffs.shape[1] != N:
        raise ValueError("Коэффициенты должны быть в квадратном блоке.")

    T = _build_dct_matrix(N)
    C = np.array([_c_factor(i) for i in range(N)])
    scale = np.outer(C, C)

    S_prime = scale * dct_coeffs
    restored = T.T @ S_prime @ T

    return 0.25 * restored
