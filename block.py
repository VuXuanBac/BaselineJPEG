import numpy as np
import cv2

from utils import ZigZagOrder
from table import QuantizationTable

class Block(object):
    def __init__(self, data: np.ndarray = None) -> None:
        self.raw = data

    def fdct(self) -> 'Block':
        self.raw = cv2.dct(np.float32(self.raw))
        return self

    def idct(self) -> 'Block':
        self.raw = cv2.idct(np.float32(self.raw))
        return self

    def dequantize(self, quantization: QuantizationTable, item_type = np.float32) -> 'Block':
        self.raw = np.array(self.raw * quantization.table, dtype=item_type)
        return self
    
    def quantize(self, quantization: QuantizationTable, item_type = np.int32) -> 'Block':
        self.raw = np.array(self.raw / quantization.table, dtype=item_type)
        return self

    def get_dc(self):
        return int(self.raw[0, 0])

    def get_copy(self, item_type = np.int32) -> np.ndarray:
        return np.array(self.raw, dtype=item_type)
    
    def get_data(self) -> np.ndarray:
        return self.raw
        
    def tolist_zigzag(self, item_type_converter = int) -> list:
        '''
        Convert 2D data (8 x 8) to 1D data (64) using zigzag order.
        :param: [item_type_converter] Element type converter for the result.
        '''
        size = 8
        result = [0] * (size * size)
        for r in range(size):
            for c in range(size):
                result[ZigZagOrder[r][c]] = item_type_converter(self.raw[r][c])
        return result

    def fromlist_zigzag(self, data: list, item_type = np.int32) -> 'Block':
        '''
        Convert 1D data (64) to 2D data (8 x 8) using zigzag order.
        :param: [item_type] Element type for the result.
        '''
        size    = 8
        result  = [[0] * size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                result[r][c] = data[ZigZagOrder[r][c]]
        self.raw = np.array(result, dtype=item_type)
        return self

class BlockExtend(object):
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

    def get_next(self) -> Block:
        indices = self.move_next()
        return Block(self.raw[indices])
    
    def get_all(self) -> np.ndarray:
        return self.raw
