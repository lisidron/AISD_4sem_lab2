import io
# Функции кодирования/декодирования блоков
from vli import get_vli_category_and_value, decode_vli


# Стандартные таблицы Хаффмана для JPEG
DEFAULT_DC_LUMINANCE_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
DEFAULT_DC_LUMINANCE_HUFFVAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

DEFAULT_DC_CHROMINANCE_BITS = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
DEFAULT_DC_CHROMINANCE_HUFFVAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

DEFAULT_AC_LUMINANCE_BITS = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
DEFAULT_AC_LUMINANCE_HUFFVAL = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]

DEFAULT_AC_CHROMINANCE_BITS = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
DEFAULT_AC_CHROMINANCE_HUFFVAL = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA
]


class HuffmanTable:
    """Генерация и использование канонических кодов Хаффмана."""
    def __init__(self, bit_counts, symbols):
        if len(bit_counts)!=16:
            raise ValueError("BITS length must be 16")
        if sum(bit_counts)!=len(symbols):
            raise ValueError("SUM(BITS) != len(HUFFVAL)")
        self.bit_counts = bit_counts[:]
        self.symbols    = symbols[:]
        self.enc_map    = {}
        self.dec_map    = {}
        self.max_len    = 0
        self._build_codes()
        self._build_decoder()

    def get_spec(self):
        return self.bit_counts, self.symbols

    def _build_codes(self):
        code = 0
        code_len = 1
        sym_idx = 0
        total = 0
        for i, cnt in enumerate(self.bit_counts):
            for _ in range(cnt):
                sym = self.symbols[sym_idx]
                self.enc_map[sym] = (code, code_len)
                sym_idx += 1
                code += 1
            total += cnt
            code <<= 1
            code_len += 1
            if cnt>0:
                self.max_len = i+1
        if total != len(self.symbols):
            print(f"Warning: generated {total} codes, expected {len(self.symbols)}")

    def _build_decoder(self):
        for sym,(c,l) in self.enc_map.items():
            b = format(c, f'0{l}b')
            self.dec_map[b] = sym

    def encode(self, symbol):
        return self.enc_map.get(symbol)

    def decode(self, reader):
        buf = ""
        for _ in range(self.max_len):
            bit = reader.read_bit()
            if bit is None: return None
            buf += str(bit)
            if buf in self.dec_map:
                return self.dec_map[buf]
        return None


class BitWriter:
    """Запись битов с байт-стаффингом JPEG."""
    def __init__(self):
        self._acc = 0
        self._pos = 0
        self._out = bytearray()

    def write_bit(self, b):
        if b not in (0,1):
            raise ValueError("Bit must be 0 or 1")
        self._acc = (self._acc<<1)|b
        self._pos += 1
        if self._pos==8:
            self._flush()

    def write_bits(self, val, n):
        if n<0: raise ValueError("n<0")
        for i in range(n-1, -1, -1):
            self.write_bit((val>>i)&1)

    def _flush(self):
        byte = self._acc & 0xFF
        self._out.append(byte)
        if byte==0xFF:
            self._out.append(0x00)
        self._acc=0
        self._pos=0

    def finish(self):
        if self._pos>0:
            pad = 8-self._pos
            self._acc = (self._acc<<pad) | ((1<<pad)-1)
            self._flush()
        return bytes(self._out)


class BitReader:
    """Чтение битов с учётом JPEG-стаффинга."""
    def __init__(self, data):
        self._stream = io.BytesIO(data)
        self._cur    = 0
        self._bit    = 8
        self._eom    = False

    def _load(self):
        if self._eom: return False
        b = self._stream.read(1)
        if not b:
            self._eom=True
            return False
        v=b[0]
        if v==0xFF:
            nxt=self._stream.read(1)
            if not nxt:
                self._eom=True
                return False
            if nxt[0]==0x00:
                self._cur=0xFF; self._bit=0
                return True
            else:
                self._stream.seek(-2,1)
                self._eom=True
                return False
        else:
            self._cur=v; self._bit=0
            return True

    def read_bit(self):
        if self._bit>7 and not self._load():
            return None
        b = (self._cur>>(7-self._bit))&1
        self._bit+=1
        return b

    def read_bits(self,n):
        if n<0: raise ValueError("n<0")
        v=0
        for _ in range(n):
            bit=self.read_bit()
            if bit is None:
                raise EOFError("Unexpected EOF")
            v=(v<<1)|bit
        return v



def huff_encode_blocks(units, dc_tbl, ac_tbl):
    writer = BitWriter()
    for dc_cat, dc_bits, ac_pairs in units:
        code, length = dc_tbl.encode(dc_cat) or (None,None)
        if code is None:
            raise ValueError(f"DC symbol {dc_cat} not in table")
        writer.write_bits(code, length)
        if dc_cat>0:
            val = int(dc_bits,2)
            writer.write_bits(val, dc_cat)
        for run, val in ac_pairs:
            if (run,val)==(0,0):
                sym=0x00
                c,l = ac_tbl.encode(sym)
                writer.write_bits(c,l)
                break
            if (run,val)==(15,0):
                sym=0xF0
                c,l = ac_tbl.encode(sym)
                writer.write_bits(c,l)
            else:
                cat, bits = get_vli_category_and_value(val)
                sym = (run<<4)|cat
                c,l = ac_tbl.encode(sym) or (None,None)
                if c is None:
                    raise ValueError(f"AC symbol {sym:02X} not in table")
                writer.write_bits(c,l)
                writer.write_bits(int(bits,2), cat)
    return writer.finish()


def huff_decode_blocks(data, dc_tbl, ac_tbl, blocks):
    reader = BitReader(data)
    out=[]
    for i in range(blocks):
        dc_cat = dc_tbl.decode(reader)
        if dc_cat is None:
            raise EOFError(f"DC at block {i+1}")
        dc_bits=""
        if dc_cat>0:
            v = reader.read_bits(dc_cat)
            dc_bits = format(v, f'0{dc_cat}b')
        ac_list=[]
        cnt=0
        while cnt<64:
            sym = ac_tbl.decode(reader)
            if sym is None:
                raise EOFError(f"AC at blk {i+1}")
            if sym==0x00:
                ac_list.append((0,0)); break
            if sym==0xF0:
                ac_list.append((15,0)); cnt+=16; continue
            run=sym>>4; cat=sym&0xF
            v=reader.read_bits(cat)
            bits=format(v,f'0{cat}b')
            val=decode_vli(cat,bits)
            ac_list.append((run,val))
            cnt+=run+1
            if cnt>63:
                print(f"Warn: ACcount {cnt}>63 blk{i+1}")
        out.append((dc_cat,dc_bits,ac_list))
    return out
