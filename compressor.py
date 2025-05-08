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

try:
    import constants
except ImportError:
    class DefaultConstants:
        Bites_for_param = 4
        ByteOrder = 'big'
        Channels = 3
    constants = DefaultConstants()
    print("Предупреждение: Файл constants.py не найден, используются значения по умолчанию.", file=sys.stderr)


BASE_Q_LUMINANCE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.uint8)

BASE_Q_CHROMINANCE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.uint8)

def save_compressed_data(filepath, metadata, y_data, cb_data, cr_data):
    """Сохраняет метаданные и сжатые байтовые потоки в файл."""
    try:
        metadata_bytes = json.dumps(metadata, indent=4).encode('utf-8')
        header_len = len(metadata_bytes)

        with open(filepath, 'wb') as f:
            f.write(b'MYJPEG')
            f.write(header_len.to_bytes(constants.Bites_for_param, constants.ByteOrder))
            f.write(metadata_bytes)
            f.write(y_data)
            f.write(cb_data)
            f.write(cr_data)
        print(f"Сжатые данные сохранены в {filepath}")
        print(f"Размер метаданных: {header_len} байт")
        print(f"Размер данных Y: {len(y_data)} байт")
        print(f"Размер данных Cb: {len(cb_data)} байт")
        print(f"Размер данных Cr: {len(cr_data)} байт")
        total_size = len(b'MYJPEG') + constants.Bites_for_param + header_len + len(y_data) + len(cb_data) + len(cr_data)
        print(f"Общий размер файла: {total_size} байт")
        orig_pixels = metadata['original_width'] * metadata['original_height'] * 3
        if orig_pixels > 0:
             ratio = orig_pixels / total_size
             print(f"Степень сжатия (приближенно): {ratio:.2f}x")

    except IOError as e:
        print(f"Ошибка записи файла {filepath}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Неожиданная ошибка при сохранении файла: {e}", file=sys.stderr)

def compress_image(image_path, output_path, quality=75, block_size=8):
    """
    Выполняет сжатие изображения из стандартного формата (PNG, BMP, и т.д.)
    по алгоритму, похожему на JPEG Baseline.
    """
    print(f"Начало сжатия '{image_path}' с качеством {quality}...")

    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
             print(f"Конвертация изображения из режима '{img.mode}' в 'RGB'...")
             img = img.convert('RGB')

        img_rgb = np.array(img)
        original_height, original_width, num_channels = img_rgb.shape

        if num_channels != 3:
            raise ValueError(f"Ожидалось 3 канала RGB, получено {num_channels}")

        print(f"Исходный размер: {original_width}x{original_height}")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден {image_path}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Ошибка при чтении или подготовке изображения: {e}", file=sys.stderr)
        return

    print("Преобразование в YCbCr...")
    img_ycbcr = ycbcr.rgb_to_ycbcr(img_rgb)
    y_channel  = img_ycbcr[:, :, 0]
    cb_channel = img_ycbcr[:, :, 1]
    cr_channel = img_ycbcr[:, :, 2]

    print("Даунсэмплинг Cb и Cr...")
    cb_downsampled = downsampling.downsample_channel_420(cb_channel)
    cr_downsampled = downsampling.downsample_channel_420(cr_channel)
    print(f"  Размер Y : {y_channel.shape}")
    print(f"  Размер Cb (DS): {cb_downsampled.shape}")
    print(f"  Размер Cr (DS): {cr_downsampled.shape}")

    print("Подготовка таблиц квантования и Хаффмана...")
    q_matrix_y = quantization.adjust_quantization_matrix(BASE_Q_LUMINANCE, quality)
    q_matrix_c = quantization.adjust_quantization_matrix(BASE_Q_CHROMINANCE, quality)

    try:
        huff_dc_y = huffman.HuffmanTable(huffman.DEFAULT_DC_LUMINANCE_BITS, huffman.DEFAULT_DC_LUMINANCE_HUFFVAL)
        huff_ac_y = huffman.HuffmanTable(huffman.DEFAULT_AC_LUMINANCE_BITS, huffman.DEFAULT_AC_LUMINANCE_HUFFVAL)
        huff_dc_c = huffman.HuffmanTable(huffman.DEFAULT_DC_CHROMINANCE_BITS, huffman.DEFAULT_DC_CHROMINANCE_HUFFVAL)
        huff_ac_c = huffman.HuffmanTable(huffman.DEFAULT_AC_CHROMINANCE_BITS, huffman.DEFAULT_AC_CHROMINANCE_HUFFVAL)
    except ValueError as e:
         print(f"Ошибка при создании таблиц Хаффмана из стандартных спецификаций: {e}", file=sys.stderr)
         return

    components_data = {}
    padded_dims = {}
    num_blocks_total = 0

    try:
        for name, channel, q_matrix, dc_table, ac_table in [
            ('Y', y_channel, q_matrix_y, huff_dc_y, huff_ac_y),
            ('Cb', cb_downsampled, q_matrix_c, huff_dc_c, huff_ac_c),
            ('Cr', cr_downsampled, q_matrix_c, huff_dc_c, huff_ac_c)
        ]:
            print(f"Обработка компонента {name}...")
            h_orig, w_orig = channel.shape
            h_pad = math.ceil(h_orig / block_size) * block_size
            w_pad = math.ceil(w_orig / block_size) * block_size
            padded_dims[name] = (h_pad, w_pad)

            blocks = tiling.split_into_blocks(channel, block_size, fill_value=128)
            num_blocks_comp = len(blocks)
            num_blocks_total += num_blocks_comp
            print(f"  {name}: {num_blocks_comp} блоков ({block_size}x{block_size})")

            quantized_blocks_data = []
            all_dc_coeffs = []

            for i, block in enumerate(blocks):
                block_shifted = block.astype(np.float64) - 128.0
                dct_coeffs = dct.dct2(block_shifted)
                quantized_coeffs = quantization.quantize(dct_coeffs, q_matrix)
                all_dc_coeffs.append(quantized_coeffs[0, 0])
                ac_coeffs_flat = zigzag.zigzag_scan(quantized_coeffs)[1:]
                ac_rle = rle_ac.prepare_ac_coefficients_for_rle(ac_coeffs_flat.tolist())
                quantized_blocks_data.append([None, None, ac_rle])

            dc_diffs = differential_dc.dpcm_encode_dc(all_dc_coeffs)
            for i, dc_diff in enumerate(dc_diffs):
                dc_category, dc_vli_bits = vli.get_vli_category_and_value(dc_diff)
                quantized_blocks_data[i][0] = dc_category
                quantized_blocks_data[i][1] = dc_vli_bits

            print(f"  Кодирование Хаффмана для {name}...")
            compressed_data = huffman.huff_encode_blocks(quantized_blocks_data, dc_table, ac_table)
            components_data[name] = compressed_data
            print(f"    Размер сжатых данных {name}: {len(compressed_data)} байт")

    except Exception as e:
        print(f"Ошибка на этапе обработки блока или кодирования Хаффмана: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    metadata = {
        "original_width": original_width,
        "original_height": original_height,
        "block_size": block_size,
        "quality": quality,
        "padded_dims_y": padded_dims['Y'],
        "padded_dims_cb": padded_dims['Cb'],
        "padded_dims_cr": padded_dims['Cr'],
        "q_table_y": q_matrix_y.tolist(),
        "q_table_c": q_matrix_c.tolist(),
        "huff_dc_y_bits": huff_dc_y.bit_counts,
        "huff_dc_y_huffval": huff_dc_y.symbols,
        "huff_ac_y_bits": huff_ac_y.bit_counts,
        "huff_ac_y_huffval": huff_ac_y.symbols,
        "huff_dc_c_bits": huff_dc_c.bit_counts,
        "huff_dc_c_huffval": huff_dc_c.symbols,
        "huff_ac_c_bits": huff_ac_c.bit_counts,
        "huff_ac_c_huffval": huff_ac_c.symbols,
        "data_len_y": len(components_data['Y']),
        "data_len_cb": len(components_data['Cb']),
        "data_len_cr": len(components_data['Cr']),
    }

    print("Сохранение результата...")
    save_compressed_data(
        output_path,
        metadata,
        components_data['Y'],
        components_data['Cb'],
        components_data['Cr']
    )

    print(f"Сжатие '{image_path}' завершено. Результат в '{output_path}'.")