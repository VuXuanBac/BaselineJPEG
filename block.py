import numpy as np
import cv2
from bitarray import bitarray

from table import QuantizationTable, HuffmanTable
from huffman import HuffmanEncoder, HuffmanDecoder
from bitutils import StateStream

class Block(object):
    '''
    Codecs for each Block (8 x 8 data)
    '''
    def __init__(self, dc_huff: HuffmanTable, ac_huff: HuffmanTable, quant: QuantizationTable, mode = 'encode') -> None:
        self.quant = quant
        self.huffman = HuffmanEncoder(dc_huff, ac_huff) if mode == 'encode' else HuffmanDecoder(dc_huff, ac_huff)
    
    def encode(self, data: np.ndarray, pred_dc: int) -> tuple[bitarray, int]:
        fdct = cv2.dct(np.float32(data))
        quant = np.int32(fdct / self.quant.table)
        return self.huffman.encode(quant, pred = pred_dc), int(quant[0][0])
    
    def decode(self, stream: StateStream, pred_dc: int) -> tuple[np.ndarray, int]:
        dec     = self.huffman.decode(stream, pred = pred_dc)
        dequant = np.float32(dec * self.quant.table)
        idct    = cv2.idct(dequant)
        return idct, int(dec[0][0])

class BlockExtend(object):
    '''
    Block Container for Rearrange Blocks.
    '''
    def __init__(self, step = (1, 1)) -> None:
        self.step           = step # ver x hor
        self.block_index    = 0
        self.group_index    = 0
        self.group_size     = step[0] * step[1]

    def end(self):
        return self.group_index >= self.group_count

    def move_next(self):
        # Extract indices for current block
        group_pos = divmod(self.group_index, self.group_step)
        block_pos = divmod(self.block_index, self.step[0])
        hor = (group_pos[0] * self.step[0] + block_pos[0]) << 3
        ver = (group_pos[1] * self.step[1] + block_pos[1]) << 3
        indices = (slice(ver, ver + 8), slice(hor, hor + 8))
        # Move to next block
        self.block_index += 1
        self.block_index %= self.group_size
        if self.block_index == 0: # new group
            self.group_index += 1
        return indices

    def build(self, size, item_type = np.int32):
        if size[0] % (8 * self.step[0]) or size[1] % (8 * self.step[1]):
            raise Exception(f'Invalid size for group of block with step {self.step}')
        self.raw = np.zeros(size, dtype=item_type)
        self.group_step = (size[0] // self.step[0]) >> 3
        self.group_count = (size[0] * size[1] // self.group_size) >> 6
        return self

    def feed(self, data: np.ndarray):
        size = data.shape
        if size[0] % (8 * self.step[0]) or size[1] % (8 * self.step[1]):
            raise Exception(f'Invalid size for group of block with step {self.step}')
        self.raw = data
        self.group_step = (size[0] // self.step[0]) >> 3
        self.group_count = (size[0] * size[1] // self.group_size) >> 6
        return self

    def put_next(self, data: np.ndarray) -> 'BlockExtend':
        indices = self.move_next()
        self.raw[indices] = data
        return self

    def get_next(self) -> np.ndarray:
        indices = self.move_next()
        return self.raw[indices] ### Change here
    
    def get_all(self) -> np.ndarray:
        return self.raw
