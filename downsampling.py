import numpy as np
import math

def downsample_channel_420(src):
    if type(src) is not np.ndarray:
        raise TypeError("Ожидался np.ndarray на входе.")
    if src.ndim != 2:
        raise ValueError("Матрица должна быть двумерной.")

    h, w = src.shape
    h2, w2 = (math.ceil(h / 2), math.ceil(w / 2))

    output = np.zeros((h2, w2), dtype=np.float32)

    for y in range(h2):
        for x in range(w2):
            y0, x0 = y * 2, x * 2
            block = src[y0 : min(y0 + 2, h), x0 : min(x0 + 2, w)]
            output[y, x] = np.mean(block) if block.size else 0

    return np.round(output).astype(np.uint8)


def upsample_channel_nearest_neighbor(input_channel, out_h, out_w):
    if not isinstance(input_channel, np.ndarray):
        raise TypeError("Вход должен быть NumPy-массивом.")
    if input_channel.ndim != 2:
        raise ValueError("Ожидалась двумерная матрица.")

    if input_channel.size == 0:
        print(f"Пустой ввод — возвращаем нулевую матрицу {out_h}x{out_w}.")
        return np.zeros((out_h, out_w), dtype=np.uint8)

    enlarged = np.repeat(np.repeat(input_channel, 2, axis=0), 2, axis=1)

    trimmed = enlarged[:out_h, :out_w]
    result = np.zeros((out_h, out_w), dtype=np.uint8)
    result[:trimmed.shape[0], :trimmed.shape[1]] = trimmed

    return result
