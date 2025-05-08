import numpy as np

def rgb_to_ycbcr(image_rgb):
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("Входное изображение должно быть массивом NumPy.")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Входное изображение должно иметь форму (height, width, 3).")
    if image_rgb.dtype != np.uint8 and (np.any(image_rgb < 0) or np.any(image_rgb > 255)):
        print("Предупреждение: значения RGB вне диапазона [0, 255]. Будут применены ограничения.")

    image_rgb_float = image_rgb.astype(np.float32)

    channel_r = image_rgb_float[:, :, 0]
    channel_g = image_rgb_float[:, :, 1]
    channel_b = image_rgb_float[:, :, 2]

    channel_y  = 0.299 * channel_r + 0.587 * channel_g + 0.114 * channel_b
    channel_cb = -0.168736 * channel_r - 0.331264 * channel_g + 0.5 * channel_b + 128.0
    channel_cr = 0.5 * channel_r - 0.418688 * channel_g - 0.081312 * channel_b + 128.0

    image_ycbcr = np.zeros_like(image_rgb_float)
    image_ycbcr[:, :, 0] = channel_y
    image_ycbcr[:, :, 1] = channel_cb
    image_ycbcr[:, :, 2] = channel_cr

    image_ycbcr = np.clip(image_ycbcr, 0, 255)

    return image_ycbcr.astype(np.uint8)

def ycbcr_to_rgb(image_ycbcr):

    if not isinstance(image_ycbcr, np.ndarray):
        raise TypeError("Входное изображение должно быть массивом NumPy.")
    if image_ycbcr.ndim != 3 or image_ycbcr.shape[2] != 3:
        raise ValueError("Входное изображение должно иметь форму (height, width, 3).")

    image_ycbcr_float = image_ycbcr.astype(np.float32)

    channel_y  = image_ycbcr_float[:, :, 0]
    channel_cb = image_ycbcr_float[:, :, 1]
    channel_cr = image_ycbcr_float[:, :, 2]

    channel_r = channel_y + 1.402 * (channel_cr - 128.0)
    channel_g = channel_y - 0.344136 * (channel_cb - 128.0) - 0.714136 * (channel_cr - 128.0)
    channel_b = channel_y + 1.772 * (channel_cb - 128.0)

    image_rgb = np.zeros_like(image_ycbcr_float)
    image_rgb[:, :, 0] = channel_r
    image_rgb[:, :, 1] = channel_g
    image_rgb[:, :, 2] = channel_b

    image_rgb = np.clip(image_rgb, 0, 255)

    return image_rgb.astype(np.uint8)
