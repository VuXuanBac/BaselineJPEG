#################################### ENCODE
########## DC ###########
# 1. Calculate DIFF.
# 2. Calculate bitsize for DIFF. [S]: minimum number of bits to represent abs(DIFF)
#       use: DIFF.bit_length()
# 3. Take S LSB of DIFF (>0) or ~(-DIFF) or 2complement(DIFF - 1) (<0) [V]
# 4. Code = [Code for S][V]
########## AC ############
# 1. Calculate number of zero preceding a non-zero number. [R]
# 2. If R > 15:
#       Code = 0xF0         # ZRL
# 3. If all remain are zero:
#       Code = 0x00         # EOB
# 4. Calculate bitsize for AC coef. [S]
# 5. Take S LSB of AC (>0) or ~AC | 2complement(AC - 1) (<0) [V]
# 6. Code = [Code for (R << 4) ^ S][V]
##########################
# Code for S and Code for RS use Huffman Table
# Huffman Table can be represented by BITS and HUFFVALS array.
#################################### DECODE
# Read bit by bit from bitstream to get a Huffman Code.
# Interprete the code. -> Get bitsize S and zero runlength R
# Read next S bits to get DIFF or AC coef

from bitarray import bitarray, util
import numpy as np
from utils import fromlist_zigzag, tolist_zigzag

from bitutils import StateStream, JpegBitConverter
from table import HuffmanTable

def gencode(bits: list[int]) -> list[bitarray]:
    '''
    From [bits] that represent the table, generate the codes for the symbols in [huffvals]
    '''
    codes = []
    value = 0
    for size, count in enumerate(bits, 1):
        for i in range(count):
            codes.append(util.int2ba(value, size))
            value += 1
        value <<= 1
    return codes

def gencode_extend(bits: list[int]):
    '''
    From [bits] that represent the table, generate data used for decoding.
    [mincode]: `mincode[S]` is the minimum code that has size (bit length) S.
    [maxcode]: `maxcode[S]` is the maximum code that has size (bit length) S.
    [mincode_index]: `mincode_index[S]` is the index of symbol in [huffvals] that has code `mincode[S]`
    '''
    maxcode = [-1] * 16
    mincode = [-1] * 16
    mincode_index = [-1] * 16
    start = 0
    value = 0
    for size, count in enumerate(bits):
        if count > 0:
            mincode_index[size] = start
            mincode[size] = value
            value += count
            start += count
            maxcode[size] = value - 1
        value <<= 1
    return mincode, maxcode, mincode_index

class Encoder(object):
    def __init__(self) -> None:
        pass

    def encode(self, data: np.ndarray, **params) -> bitarray:
        pass

class Decoder(object):
    def __init__(self) -> None:
        pass

    def decode(self, stream: StateStream, **params) -> np.ndarray:
        pass

class _EncoderLookup(object):
    def __init__(self, table: HuffmanTable) -> None:
        codes = gencode(table.bits)
        symbol_code = {}
        for index, symbol in enumerate(table.symbols):
            symbol_code[symbol] = codes[index]

        self.mappings = symbol_code

    def get_code(self, symbol: int) -> bitarray:
        return self.mappings[symbol]

class _DecoderLookup(object):
    def __init__(self, table: HuffmanTable) -> None:
        self.mincode, self.maxcode, self.mincode_index = gencode_extend(table.bits)
        self.symbols = table.symbols

    def get_symbol(self, stream: StateStream) -> int:
        '''
        Read from [stream] bit by bit until get a correct code.
        :return: The symbol for the code.
        '''
        code_size = 0 # number of read bits = bit size of code
        code = stream.next_bit()

        while code > self.maxcode[code_size]:
            code = (code << 1) + stream.next_bit()
            code_size += 1
        code_index = self.mincode_index[code_size] + code - self.mincode[code_size]

        return self.symbols[code_index]

class HuffmanEncoder(Encoder):
    def __init__(self, dc_table: HuffmanTable, ac_table: HuffmanTable) -> None:
        super().__init__()
        self.dc_lookup = _EncoderLookup(dc_table)
        self.ac_lookup = _EncoderLookup(ac_table)

    def encode(self, data: np.ndarray, **params) -> bitarray:
        pred = params['pred']
        coefs = tolist_zigzag(data) # data.tolist_zigzag() ### Change here
        result = bitarray()
        ######## Encode DC ########
        diff = coefs[0] - pred
        category = diff.bit_length() # number of bits represent the diff = 
        # if category > self.max_dc_category:
        #     raise Exception(f'Invalid DC coef')
        result.extend(self.dc_lookup.get_code(category))
        if category > 0:
            result.extend(JpegBitConverter.int2bits(diff))
        ######## Encode AC ########
        zrl = 0 # run length of zeros
        for ac in coefs[1:]:
            if ac == 0:
                zrl += 1
            else:
                ### process run length ###
                while zrl > 15:
                    result.extend(self.ac_lookup.get_code(0xF0))
                    zrl -= 16
                category = ac.bit_length()
                # if category > self.max_ac_category:
                #     raise Exception(f'Invalid AC coef')
                rs = (zrl << 4) | category
                result.extend(self.ac_lookup.get_code(rs))
                result.extend(JpegBitConverter.int2bits(ac))
                zrl = 0
        if zrl > 0: # EOB
            result.extend(self.ac_lookup.get_code(0x00))
        return result

class HuffmanDecoder(Decoder):
    def __init__(self, dc_table: HuffmanTable, ac_table: HuffmanTable) -> None:
        self.dc_lookup = _DecoderLookup(dc_table)
        self.ac_lookup = _DecoderLookup(ac_table)
    
    def decode(self, stream: StateStream, **params) -> np.ndarray:
        number_of_coefs     = 64
        coefs               = [0] * number_of_coefs
        pred                = params['pred']
        ######## Decode DC ########
        precision = self.dc_lookup.get_symbol(stream) # number of bits represent the diff
        diff = 0 if precision == 0 else JpegBitConverter.bits2int(stream.next_bits(precision))
        coefs[0] = pred + diff
        ######## Decode AC ########
        k = 1 # index in coefs
        while k < number_of_coefs:
            rs = self.ac_lookup.get_symbol(stream) # run length || number of bits represent the coef
            zrl = rs >> 4
            precision = rs & 0x0F
            k += zrl
            if precision > 0:
                coefs[k] = JpegBitConverter.bits2int(stream.next_bits(precision))
            elif zrl != 0xF: # Not zero runlength
                break
            k += 1

        return fromlist_zigzag(coefs)# Block(None).fromlist_zigzag(coefs) ### Change here