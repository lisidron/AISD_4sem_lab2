import numpy as np

def split_into_blocks(image_channel, block_size, fill_value=0):

    if not isinstance(image_channel, np.ndarray):
        raise TypeError("Ожидается массив NumPy.")
    if image_channel.ndim != 2:
        raise ValueError("Матрица должна быть двумерной.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Размер блока должен быть положительным целым числом.")

    h, w = image_channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size

    padded = np.pad(image_channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=fill_value)

    blocks = []
    for i in range(0, padded.shape[0], block_size):
        for j in range(0, padded.shape[1], block_size):
            blocks.append(padded[i:i + block_size, j:j + block_size])

    return blocks

def assemble_from_blocks(blocks_list, padded_height, padded_width, original_height=None, original_width=None):
    if not blocks_list:
        return np.zeros((0, 0), dtype=np.uint8)

    block_shape = blocks_list[0].shape
    if block_shape[0] != block_shape[1] or block_shape[0] == 0:
        raise ValueError("Блоки должны быть квадратными и ненулевого размера.")
    block_size = block_shape[0]

    if padded_height % block_size != 0 or padded_width % block_size != 0:
        raise ValueError("Размеры padded_height и padded_width должны быть кратны размеру блока.")

    num_blocks_h = padded_height // block_size
    num_blocks_w = padded_width // block_size

    expected_blocks = num_blocks_h * num_blocks_w
    if len(blocks_list) != expected_blocks:
        raise ValueError(f"Ожидалось {expected_blocks} блоков, получено {len(blocks_list)}.")

    reassembled = np.zeros((padded_height, padded_width), dtype=blocks_list[0].dtype)

    index = 0
    for r in range(num_blocks_h):
        for c in range(num_blocks_w):
            start_r = r * block_size
            end_r = start_r + block_size
            start_c = c * block_size
            end_c = start_c + block_size

            block = blocks_list[index]
            if block.shape != (block_size, block_size):
                raise ValueError(f"Неверный размер блока {index}: {block.shape}, ожидался {(block_size, block_size)}")

            reassembled[start_r:end_r, start_c:end_c] = block
            index += 1

    if original_height is not None and original_width is not None:
        return reassembled[:original_height, :original_width]

    return reassembled
