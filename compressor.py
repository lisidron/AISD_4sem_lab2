import numpy as np
from PIL import Image
import json
import math
import sys

import ycbcr
import downsampling
import tiling
import dct
import quantization
import zigzag
import differential_dc
import rle_ac
import vli
import huffman

# Загрузка настроек по умолчанию
try:
    import constants
except ImportError:
    class FallbackConstants:
        PARAM_BYTES = 4
        BYTE_ORDER = 'big'
        COLOR_CHANNELS = 3
    constants = FallbackConstants()
    print("Не найден constants.py — применяются значения по умолчанию", file=sys.stderr)

LUMA_QTABLE = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)
CHROMA_QTABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.uint8)


def write_encoded_file(filename, meta, y_bytes, cb_bytes, cr_bytes):
    try:
        header_json = json.dumps(meta, indent=4).encode('utf-8')
        header_size = len(header_json)

        with open(filename, 'wb') as file:
            file.write(b'MYJPEG')
            file.write(header_size.to_bytes(constants.PARAM_BYTES, constants.BYTE_ORDER))
            file.write(header_json)
            file.write(y_bytes)
            file.write(cb_bytes)
            file.write(cr_bytes)

        print(f"\n Файл сохранён: {filename}")
        print(f" Размер заголовка: {header_size} байт")
        print(f" Y: {len(y_bytes)} байт, Cb: {len(cb_bytes)} байт, Cr: {len(cr_bytes)} байт")

        original_pixels = meta['original_width'] * meta['original_height'] * 3
        final_size = sum(map(len, [b'MYJPEG', y_bytes, cb_bytes, cr_bytes])) + constants.PARAM_BYTES + header_size
        compression_ratio = original_pixels / final_size if final_size > 0 else 0
        print(f" Сжатие: ~{compression_ratio:.2f}x")

    except IOError as io_err:
        print(f" Ошибка записи: {io_err}", file=sys.stderr)
    except Exception as ex:
        print(f" Неожиданная ошибка: {ex}", file=sys.stderr)


def jpeg_like_compression(input_image, output_file, quality=75, tile_size=8):
    print(f"\n Старт обработки: {input_image} (качество: {quality})")

    try:
        with Image.open(input_image) as im:
            if im.mode != 'RGB':
                print(f" Перевод из {im.mode} в RGB")
                im = im.convert('RGB')
            rgb_array = np.array(im)

    except FileNotFoundError:
        print(f" Файл не найден: {input_image}", file=sys.stderr)
        return
    except Exception as err:
        print(f" Ошибка при загрузке: {err}", file=sys.stderr)
        return

    height, width, channels = rgb_array.shape
    if channels != 3:
        print(f" Ожидалось 3 канала RGB, получено: {channels}", file=sys.stderr)
        return

    print(f" Размер: {width}x{height}")

    # Преобразование цвета
    ycbcr_img = ycbcr.rgb_to_ycbcr(rgb_array)
    y_plane, cb_plane, cr_plane = ycbcr_img[:, :, 0], ycbcr_img[:, :, 1], ycbcr_img[:, :, 2]

    print(" Даунсэмплирование...")
    cb_ds = downsampling.downsample_channel_420(cb_plane)
    cr_ds = downsampling.downsample_channel_420(cr_plane)

    print(" Подготовка матриц и таблиц...")
    q_y = quantization.adjust_quantization_matrix(LUMA_QTABLE, quality)
    q_c = quantization.adjust_quantization_matrix(CHROMA_QTABLE, quality)

    try:
        huff_tables = {
            "Y_DC": huffman.HuffmanTable(huffman.DEFAULT_DC_LUMINANCE_BITS, huffman.DEFAULT_DC_LUMINANCE_HUFFVAL),
            "Y_AC": huffman.HuffmanTable(huffman.DEFAULT_AC_LUMINANCE_BITS, huffman.DEFAULT_AC_LUMINANCE_HUFFVAL),
            "C_DC": huffman.HuffmanTable(huffman.DEFAULT_DC_CHROMINANCE_BITS, huffman.DEFAULT_DC_CHROMINANCE_HUFFVAL),
            "C_AC": huffman.HuffmanTable(huffman.DEFAULT_AC_CHROMINANCE_BITS, huffman.DEFAULT_AC_CHROMINANCE_HUFFVAL),
        }
    except ValueError as err:
        print(f" Ошибка инициализации Хаффмана: {err}", file=sys.stderr)
        return

    # Обработка всех компонент
    all_encoded = {}
    padded_sizes = {}

    for label, channel, q_matrix, dc_huff, ac_huff in [
        ('Y', y_plane, q_y, huff_tables["Y_DC"], huff_tables["Y_AC"]),
        ('Cb', cb_ds, q_c, huff_tables["C_DC"], huff_tables["C_AC"]),
        ('Cr', cr_ds, q_c, huff_tables["C_DC"], huff_tables["C_AC"]),
    ]:
        print(f"\n Компонент: {label}")
        h, w = channel.shape
        h_padded = math.ceil(h / tile_size) * tile_size
        w_padded = math.ceil(w / tile_size) * tile_size
        padded_sizes[label] = (h_padded, w_padded)

        blocks = tiling.split_into_blocks(channel, tile_size, fill_value=128)
        dct_blocks = []
        dc_values = []

        for blk in blocks:
            shifted = blk.astype(np.float64) - 128.0
            transformed = dct.dct2(shifted)
            quantized = quantization.quantize(transformed, q_matrix)
            dc_values.append(quantized[0, 0])

            zz = zigzag.zigzag_scan(quantized)[1:]
            ac_encoded = rle_ac.prepare_ac_coefficients_for_rle(zz.tolist())
            dct_blocks.append([None, None, ac_encoded])

        dc_deltas = differential_dc.dpcm_encode_dc(dc_values)
        for i, delta in enumerate(dc_deltas):
            category, bits = vli.get_vli_category_and_value(delta)
            dct_blocks[i][0] = category
            dct_blocks[i][1] = bits

        print(f" Хаффман {label}")
        encoded_stream = huffman.huff_encode_blocks(dct_blocks, dc_huff, ac_huff)
        all_encoded[label] = encoded_stream
        print(f" {label} готово: {len(encoded_stream)} байт")

    # Метаданные
    image_info = {
        "original_width": width,
        "original_height": height,
        "block_size": tile_size,
        "quality": quality,
        "padded_dims_y": padded_sizes['Y'],
        "padded_dims_cb": padded_sizes['Cb'],
        "padded_dims_cr": padded_sizes['Cr'],
        "q_table_y": q_y.tolist(),
        "q_table_c": q_c.tolist(),
        "huff_dc_y_bits": huff_tables["Y_DC"].bit_counts,
        "huff_dc_y_huffval": huff_tables["Y_DC"].symbols,
        "huff_ac_y_bits": huff_tables["Y_AC"].bit_counts,
        "huff_ac_y_huffval": huff_tables["Y_AC"].symbols,
        "huff_dc_c_bits": huff_tables["C_DC"].bit_counts,
        "huff_dc_c_huffval": huff_tables["C_DC"].symbols,
        "huff_ac_c_bits": huff_tables["C_AC"].bit_counts,
        "huff_ac_c_huffval": huff_tables["C_AC"].symbols,
        "data_len_y": len(all_encoded['Y']),
        "data_len_cb": len(all_encoded['Cb']),
        "data_len_cr": len(all_encoded['Cr']),
    }

    print("\n Сохранение...")
    write_encoded_file(
        output_file,
        image_info,
        all_encoded['Y'],
        all_encoded['Cb'],
        all_encoded['Cr']
    )

    print(f"\n Завершено: {input_image} - {output_file}")
