import numpy as np


def prepare_ac_coefficients_for_rle(ac_coeffs_zigzag):
    if not isinstance(ac_coeffs_zigzag, (list, np.ndarray)):
        raise TypeError("Входные AC коэффициенты должны быть списком или массивом NumPy.")

    encoded_ac = []
    zero_run_length = 0

    for coeff in ac_coeffs_zigzag:
        if coeff == 0:
            zero_run_length += 1
            if zero_run_length == 16:
                encoded_ac.append((15, 0))
                zero_run_length = 0
        else:
            encoded_ac.append((zero_run_length, coeff))
            zero_run_length = 0

    encoded_ac.append((0, 0))  # EOB (End of Block)

    return encoded_ac

def restore_ac_coefficients_from_rle(encoded_ac, num_ac_coeffs=63):
    if not isinstance(encoded_ac, list):
        raise TypeError("Входные RLE AC коэффициенты должны быть списком.")

    ac_coeffs = [0] * num_ac_coeffs
    current_idx = 0

    for run_length, value in encoded_ac:
        if (run_length, value) == (0, 0):  # EOB
            break
        elif run_length == 15 and value == 0:  # ZRL
            if current_idx + 16 > num_ac_coeffs:
                raise ValueError("ZRL превышает размер блока")
            current_idx += 16
        else:
            if current_idx + run_length + 1 > num_ac_coeffs:
                raise ValueError("Run-length превышает размер блока")
            current_idx += run_length
            ac_coeffs[current_idx] = value
            current_idx += 1

    return ac_coeffs

