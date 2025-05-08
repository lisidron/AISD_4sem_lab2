import numpy as np
def adjust_quantization_matrix(base_matrix, quality_factor):
    if not isinstance(base_matrix, np.ndarray):
        raise TypeError("Базовая матрица должна быть массивом NumPy.")
    if not (1 <= quality_factor <= 100):
        raise ValueError("Уровень качества должен быть в диапазоне от 1 до 100.")

    # Корректируем quality_factor, если он выходит за пределы
    if quality_factor < 1:
        quality_factor = 1
    if quality_factor > 100:
        quality_factor = 100

    base_matrix_float = base_matrix.astype(np.float64)

    # Вычисляем коэффициент масштабирования в зависимости от качества
    if quality_factor < 50:
        scale_factor = 5000.0 / quality_factor
    else:
        scale_factor = 200.0 - 2.0 * quality_factor

    adjusted_matrix_float = (base_matrix_float * scale_factor + 50.0) / 100.0
    adjusted_matrix_int = np.floor(adjusted_matrix_float)

    # Ограничиваем значения в диапазоне от 1 до 255
    adjusted_matrix_int[adjusted_matrix_int < 1] = 1
    adjusted_matrix_int[adjusted_matrix_int > 255] = 255

    return adjusted_matrix_int.astype(np.uint8)
def quantize(dct_coeffs_block, quantization_matrix):
    if dct_coeffs_block.shape != quantization_matrix.shape:
        raise ValueError("Размеры блока коэффициентов и матрицы квантования должны совпадать.")
    if not np.all(quantization_matrix >= 1):
        raise ValueError("Все значения в матрице квантования должны быть >= 1.")

    # Приводим коэффициенты DCT к типу float64 для точности при делении
    dct_coeffs_block_float = dct_coeffs_block.astype(np.float64)

    # Выполняем квантувание
    quantized_coeffs = np.round(dct_coeffs_block_float / quantization_matrix.astype(np.float64))

    # Возвращаем результат в типе int32
    return quantized_coeffs.astype(np.int32)
def dequantize(quantized_coeffs_block, quantization_matrix):
    if quantized_coeffs_block.shape != quantization_matrix.shape:
        raise ValueError("Размеры блока квантованных коэффициентов и матрицы квантования должны совпадать.")

    # Приводим квантованные коэффициенты и матрицу квантования к типу float64 для точности при вычислениях
    quantized_coeffs_block_float = quantized_coeffs_block.astype(np.float64)
    quantization_matrix_float = quantization_matrix.astype(np.float64)

    # Выполняем де-квантование
    dequantized_coeffs = quantized_coeffs_block_float * quantization_matrix_float

    # Возвращаем результат в типе float64 для точности
    return dequantized_coeffs
