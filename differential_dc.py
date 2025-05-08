import numpy as np

def dpcm_encode_dc(dc_coefficients):
    if not isinstance(dc_coefficients, (list, np.ndarray)):
        raise TypeError("Входные DC коэффициенты должны быть списком или массивом NumPy.")
    if len(dc_coefficients) == 0:
        return []

    dc_coeffs = np.array(dc_coefficients, dtype=np.int32)
    diff_dc = np.empty_like(dc_coeffs)

    # Инициализация предсказания для первого коэффициента (равно 0 для первого блока в JPEG).
    pred_dc = 0
    diff_dc[0] = dc_coeffs[0] - pred_dc

    # Разностное кодирование для остальных коэффициентов.
    for i in range(1, len(dc_coeffs)):
        diff_dc[i] = dc_coeffs[i] - dc_coeffs[i - 1]

    return diff_dc.tolist()


def dpcm_decode_dc(diff_dc_coefficients):
    if not isinstance(diff_dc_coefficients, (list, np.ndarray)):
        raise TypeError("Входные разностные DC коэффициенты должны быть списком или массивом NumPy.")
    if len(diff_dc_coefficients) == 0:
        return []

    diff_dc_coeffs = np.array(diff_dc_coefficients, dtype=np.int32)
    dc_coeffs = np.empty_like(diff_dc_coeffs)

    # Восстановление первого коэффициента (предполагается, что оно равно 0).
    pred_dc = 0
    dc_coeffs[0] = diff_dc_coeffs[0] + pred_dc

    # Восстановление остальных коэффициентов.
    for i in range(1, len(diff_dc_coeffs)):
        dc_coeffs[i] = dc_coeffs[i - 1] + diff_dc_coeffs[i]

    return dc_coeffs.tolist()
