import numpy as np

class QuantizationTable(object):
    def __init__(self, data: np.ndarray | list[list]) -> None:
        self.table = np.array(data)

    def scale(self, factor: float | None) -> 'QuantizationTable':
        if factor:
            temp = np.round(self.table * factor)
            temp[temp <= 0] = 1
        else:
            temp = np.ones_like(self.table)
        return QuantizationTable(temp)

class HuffmanTable(object):
    def __init__(self, bits: list[int], huffvals: list[int]) -> None:
        self.bits       = bits
        self.symbols    = huffvals

    def frombytes(bits_huffvals: bytes):
        bits = list(bits_huffvals[:16])
        huffvals = []
        current = 16
        for b in bits:
            huffvals.extend(list(bits_huffvals[current:current + b]))
            current += b
        return HuffmanTable(bits, huffvals)
    
_luma_quant_q50 = [[16, 11, 10, 16, 24, 40, 51, 61],
                       [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56],
                       [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77],
                       [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101],
                       [72, 92, 95, 98, 112, 100, 103, 99]]

_chroma_quant_q50 = [[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]]

_luma_dc    = bytes.fromhex('00 01 05 01 01 01 01 01 01 00 00 00 00 00 00 00 00 01 02 03 04 05 06 07 08 09 0A 0B')
_chroma_dc  = bytes.fromhex('00 03 01 01 01 01 01 01 01 01 01 00 00 00 00 00 00 01 02 03 04 05 06 07 08 09 0A 0B')
_luma_ac    = bytes.fromhex('00 02 01 03 03 02 04 03 05 05 04 04 00 00 01 7D 01 02 03 00 04 11 05 12 21 31 41 06 13 51 61 07 22 71 14 32 81 91 A1 08 23 42 B1 C1 15 52 D1 F0 24 33 62 72 82 09 0A 16 17 18 19 1A 25 26 27 28 29 2A 34 35 36 37 38 39 3A 43 44 45 46 47 48 49 4A 53 54 55 56 57 58 59 5A 63 64 65 66 67 68 69 6A 73 74 75 76 77 78 79 7A 83 84 85 86 87 88 89 8A 92 93 94 95 96 97 98 99 9A A2 A3 A4 A5 A6 A7 A8 A9 AA B2 B3 B4 B5 B6 B7 B8 B9 BA C2 C3 C4 C5 C6 C7 C8 C9 CA D2 D3 D4 D5 D6 D7 D8 D9 DA E1 E2 E3 E4 E5 E6 E7 E8 E9 EA F1 F2 F3 F4 F5 F6 F7 F8 F9 FA')
_chroma_ac  = bytes.fromhex('00 02 01 02 04 04 03 04 07 05 04 04 00 01 02 77 00 01 02 03 11 04 05 21 31 06 12 41 51 07 61 71 13 22 32 81 08 14 42 91 A1 B1 C1 09 23 33 52 F0 15 62 72 D1 0A 16 24 34 E1 25 F1 17 18 19 1A 26 27 28 29 2A 35 36 37 38 39 3A 43 44 45 46 47 48 49 4A 53 54 55 56 57 58 59 5A 63 64 65 66 67 68 69 6A 73 74 75 76 77 78 79 7A 82 83 84 85 86 87 88 89 8A 92 93 94 95 96 97 98 99 9A A2 A3 A4 A5 A6 A7 A8 A9 AA B2 B3 B4 B5 B6 B7 B8 B9 BA C2 C3 C4 C5 C6 C7 C8 C9 CA D2 D3 D4 D5 D6 D7 D8 D9 DA E2 E3 E4 E5 E6 E7 E8 E9 EA F2 F3 F4 F5 F6 F7 F8 F9 FA')

def get_suggest_quant_table():
    return QuantizationTable(_luma_quant_q50), QuantizationTable(_chroma_quant_q50)

def get_suggest_luma_huffman_table():
    return HuffmanTable.frombytes(_luma_dc), HuffmanTable.frombytes(_luma_ac)

def get_suggest_chroma_huffman_table():
    return HuffmanTable.frombytes(_chroma_dc), HuffmanTable.frombytes(_chroma_ac)