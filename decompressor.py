import numpy as np
from PIL import Image
import json
import math
import sys
import os

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

_use_matplotlib = False
try:
    import raw_image_utils
    import matplotlib.pyplot as plt
    _use_matplotlib = True
except ImportError:
    raw_image_utils = None
    plt = None


def load_compressed_data(filepath):
    """Загружает метаданные и сжатые байтовые потоки из файла."""
    metadata = None
    y_data, cb_data, cr_data = None, None, None
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(len(b'MYJPEG'))
            if magic != b'MYJPEG':
                raise ValueError("Неверный формат файла (не найден 'MYJPEG')")

            header_len_bytes = f.read(constants.Bites_for_param)
            if len(header_len_bytes) != constants.Bites_for_param:
                raise EOFError("Не удалось прочитать длину заголовка.")
            header_len = int.from_bytes(header_len_bytes, constants.ByteOrder)

            metadata_bytes = f.read(header_len)
            if len(metadata_bytes) != header_len:
                raise EOFError("Не удалось прочитать полный заголовок.")
            metadata = json.loads(metadata_bytes.decode('utf-8'))

            required_keys = [
                "data_len_y", "data_len_cb", "data_len_cr", "original_width", "original_height",
                 "block_size", "q_table_y", "q_table_c", "huff_dc_y_bits", "huff_dc_y_huffval",
                 "huff_ac_y_bits", "huff_ac_y_huffval", "huff_dc_c_bits", "huff_dc_c_huffval",
                 "huff_ac_c_bits", "huff_ac_c_huffval", "padded_dims_y", "padded_dims_cb", "padded_dims_cr"
                 ]
            for key in required_keys:
                if key not in metadata:
                    raise ValueError(f"Отсутствует необходимый ключ в метаданных: {key}")

            y_data = f.read(metadata["data_len_y"])
            cb_data = f.read(metadata["data_len_cb"])
            cr_data = f.read(metadata["data_len_cr"])

            if len(y_data) != metadata["data_len_y"] or \
               len(cb_data) != metadata["data_len_cb"] or \
               len(cr_data) != metadata["data_len_cr"]:
                raise EOFError("Не удалось прочитать полные сжатые данные для компонентов.")

            print(f"Метаданные и сжатые данные успешно загружены из {filepath}")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден {filepath}", file=sys.stderr)
        return None, None, None, None
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON в метаданных: {e}", file=sys.stderr)
        return None, None, None, None
    except (ValueError, EOFError) as e:
        print(f"Ошибка чтения или формата файла {filepath}: {e}", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"Неожиданная ошибка при загрузке файла: {e}", file=sys.stderr)
        return None, None, None, None

    return metadata, y_data, cb_data, cr_data


def decompress_image(compressed_path, output_path):
    """
    Выполняет декомпрессию изображения из формата .myjpeg в стандартный формат (напр. PNG).
    """
    print(f"Начало декомпрессии '{compressed_path}'...")

    metadata, y_data, cb_data, cr_data = load_compressed_data(compressed_path)
    if metadata is None:
        return None

    try:
        print("Восстановление таблиц и параметров...")
        block_size = metadata['block_size']
        original_width = metadata['original_width']
        original_height = metadata['original_height']
        padded_dims = {
            'Y': tuple(metadata['padded_dims_y']),
            'Cb': tuple(metadata['padded_dims_cb']),
            'Cr': tuple(metadata['padded_dims_cr'])
        }

        q_matrix_y = np.array(metadata['q_table_y'], dtype=np.uint8)
        q_matrix_c = np.array(metadata['q_table_c'], dtype=np.uint8)

        huff_dc_y = huffman.HuffmanTable(metadata['huff_dc_y_bits'], metadata['huff_dc_y_huffval'])
        huff_ac_y = huffman.HuffmanTable(metadata['huff_ac_y_bits'], metadata['huff_ac_y_huffval'])
        huff_dc_c = huffman.HuffmanTable(metadata['huff_dc_c_bits'], metadata['huff_dc_c_huffval'])
        huff_ac_c = huffman.HuffmanTable(metadata['huff_ac_c_bits'], metadata['huff_ac_c_huffval'])

        reconstructed_channels = {}

        for name, comp_data, dc_table, ac_table, q_matrix in [
            ('Y', y_data, huff_dc_y, huff_ac_y, q_matrix_y),
            ('Cb', cb_data, huff_dc_c, huff_ac_c, q_matrix_c),
            ('Cr', cr_data, huff_dc_c, huff_ac_c, q_matrix_c)
        ]:
            print(f"Декодирование компонента {name}...")
            h_pad, w_pad = padded_dims[name]
            num_blocks_comp = (h_pad // block_size) * (w_pad // block_size)
            if num_blocks_comp == 0 and len(comp_data) > 0:
                 raise ValueError(f"Расчетное количество блоков 0, но есть данные для {name}")
            elif num_blocks_comp == 0 and len(comp_data) == 0:
                 reconstructed_channels[name] = np.zeros((0,0), dtype=np.uint8)
                 continue

            decoded_block_data = huffman.huff_decode_blocks(comp_data, dc_table, ac_table, num_blocks_comp)
            if len(decoded_block_data) != num_blocks_comp:
                 print(f"Предупреждение: декодировано {len(decoded_block_data)} блоков для {name}, ожидалось {num_blocks_comp}")
                 num_blocks_comp = len(decoded_block_data)
                 if num_blocks_comp == 0:
                      reconstructed_channels[name] = np.zeros((0,0), dtype=np.uint8)
                      continue

            all_dc_diffs = []
            quantized_blocks_list = []

            print(f"  Восстановление {num_blocks_comp} квантованных блоков {name}...")
            for dc_category, dc_vli_bits, ac_rle_pairs in decoded_block_data:
                ac_zigzag = rle_ac.restore_ac_coefficients_from_rle(ac_rle_pairs, block_size * block_size - 1)
                dc_diff = vli.decode_vli(dc_category, dc_vli_bits)
                all_dc_diffs.append(dc_diff)
                zigzag_flat = np.array([dc_diff] + ac_zigzag, dtype=np.int32)
                if len(zigzag_flat) != block_size * block_size:
                    raise ValueError(f"Неверная длина ({len(zigzag_flat)}) восстановленного зигзаг-массива для блока. Ожидалось {block_size*block_size}.")
                quant_block_with_dc_diff = zigzag.inverse_zigzag_scan(zigzag_flat, block_size)
                quantized_blocks_list.append(quant_block_with_dc_diff)

            print(f"  Применение обратного DPCM к DC {name}...")
            dc_actual_values = differential_dc.dpcm_decode_dc(all_dc_diffs)

            if len(dc_actual_values) != len(quantized_blocks_list):
                 raise ValueError(f"Несовпадение количества DC ({len(dc_actual_values)}) и блоков ({len(quantized_blocks_list)}) для {name}")

            final_component_blocks = []
            print(f"  Деквантование и IDCT для блоков {name}...")
            for i, quant_block in enumerate(quantized_blocks_list):
                quant_block[0, 0] = dc_actual_values[i]
                dequantized_coeffs = quantization.dequantize(quant_block, q_matrix)
                reconstructed_shifted = dct.idct2(dequantized_coeffs)
                reconstructed_leveled = reconstructed_shifted + 128.0
                reconstructed_final = np.clip(reconstructed_leveled, 0, 255)
                final_component_blocks.append(np.round(reconstructed_final).astype(np.uint8))

            print(f"  Сборка компонента {name}...")
            if not final_component_blocks:
                 reassembled_padded = np.zeros((h_pad, w_pad), dtype=np.uint8)
            else:
                 reassembled_padded = tiling.assemble_from_blocks(final_component_blocks, h_pad, w_pad)

            if name == 'Y':
                final_h, final_w = original_height, original_width
            else:
                final_h = math.ceil(original_height / 2)
                final_w = math.ceil(original_width / 2)

            final_h = min(final_h, h_pad)
            final_w = min(final_w, w_pad)

            reconstructed_channels[name] = reassembled_padded[:final_h, :final_w]
            print(f"    Финальный размер {name}: {reconstructed_channels[name].shape}")

        print("Апсэмплинг Cb и Cr...")
        y_final = reconstructed_channels['Y']
        target_h, target_w = y_final.shape

        if reconstructed_channels['Cb'].size == 0 or reconstructed_channels['Cr'].size == 0:
            print("Предупреждение: Каналы Cb/Cr пусты после декомпрессии/обрезки. Возможно, исходное изображение было < 2x2.")
            cb_upsampled = np.full((target_h, target_w), 128, dtype=np.uint8)
            cr_upsampled = np.full((target_h, target_w), 128, dtype=np.uint8)
        else:
            cb_upsampled = downsampling.upsample_channel_nearest_neighbor(reconstructed_channels['Cb'], target_h, target_w)
            cr_upsampled = downsampling.upsample_channel_nearest_neighbor(reconstructed_channels['Cr'], target_h, target_w)

        print(f"  Размер Y : {y_final.shape}")
        print(f"  Размер Cb (US): {cb_upsampled.shape}")
        print(f"  Размер Cr (US): {cr_upsampled.shape}")

        if not (y_final.shape == cb_upsampled.shape == cr_upsampled.shape):
             raise ValueError(f"Размеры каналов после апсэмплинга не совпадают: "
                              f"Y={y_final.shape}, Cb={cb_upsampled.shape}, Cr={cr_upsampled.shape}")

        final_ycbcr = np.stack((y_final, cb_upsampled, cr_upsampled), axis=-1)
        final_rgb = ycbcr.ycbcr_to_rgb(final_ycbcr)

        print(f"Сохранение восстановленного изображения в {output_path}...")
        img_out = Image.fromarray(final_rgb)
        img_out.save(output_path)

        print(f"Декомпрессия '{compressed_path}' завершена. Результат в '{output_path}'.")
        return final_rgb

    except Exception as e:
        print(f"Ошибка во время процесса декомпрессии: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def display_compressed_image(compressed_path):
    """Декомпрессирует и отображает изображение с помощью Pillow."""
    print(f"Попытка отображения сжатого изображения: {compressed_path}")
    if not os.path.exists(compressed_path):
         print(f"Ошибка: Файл не найден {compressed_path}", file=sys.stderr)
         return

    import tempfile
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_f:
            temp_output_path = temp_f.name
        print(f"Декомпрессия во временный файл: {temp_output_path}")

        decompressed_rgb = decompress_image(compressed_path, temp_output_path)

        if decompressed_rgb is not None:
            print(f"Показ изображения из {temp_output_path}...")
            img_display = Image.open(temp_output_path)
            img_display.show()

        else:
            print("Декомпрессия не удалась, отображение невозможно.")

    except Exception as e:
        print(f"Ошибка при декомпрессии или отображении: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
            try:
                os.remove(temp_output_path)
                print(f"Временный файл {temp_output_path} удален.")
            except OSError as rm_err:
                print(f"Не удалось удалить временный файл {temp_output_path}: {rm_err}", file=sys.stderr)