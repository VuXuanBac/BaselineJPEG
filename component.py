import cv2
import numpy as np

import utils
from bitutils import StateStream
from block import BlockExtend, Block

class Component(object):
    '''
    Codecs for each Component
    '''
    def __init__(self) -> None:
        self.sampling_factor    = (1, 1)
        self.interpolation      = cv2.INTER_AREA
        self.quantization_table = None
        self.huffman_tables     = None
        self.quality            = 50

    def set_quality(self, quality: int):
        self.quality = quality

    def preencode(self, data: np.ndarray, max_sfactor, mode = 'non-interleave') -> np.ndarray:
        '''
        Perform Downsampling and Padding (to proper size) on [data].
        '''
        sh, sw = utils.calculate_sampling_size(data.shape[:2], self.sampling_factor, max_sfactor)
        sampling = cv2.resize(data, (sw, sh), interpolation=self.interpolation) # down sampling
        
        eh, ew = utils.calculate_padding_size((sh, sw), self.sampling_factor, mode)
        return cv2.copyMakeBorder(sampling, 0, eh - sh, 0, ew - sw, cv2.BORDER_REPLICATE) # expand to divisible

    def encode(self, data: np.ndarray, mode: str = 'non-interleave'):
        '''
        Encode [data] and Yield one block's decoded data.
        '''
        step      = (1, 1) if mode == 'non-interleave' else (self.sampling_factor[1], self.sampling_factor[0])
        blockextend     = BlockExtend(step).feed(data)

        quanttable = self.quantization_table.scale(utils.compute_scale_factor(self.quality))
        blockencoder = Block(*self.huffman_tables, quanttable, 'encode')
        pred         = 0

        while not blockextend.end():
            benc, pred  = blockencoder.encode(blockextend.get_next(), pred)
            yield benc

    def decode(self, stream: StateStream, mode = 'non-interleave'):
        '''
        Decode [stream] and Yield one block data
        '''
        pred            = 0
        quanttable = self.quantization_table.scale(utils.compute_scale_factor(self.quality))
        blockdecoder = Block(*self.huffman_tables, quanttable, 'decode')

        while not stream.end():
            bdec, pred   = blockdecoder.decode(stream, pred)
            yield bdec

    def postdecode(self, data: np.ndarray, max_sampling_factor, original_shape) -> np.ndarray:
        '''
        Perform Cropping (remove padding) and Upsampling on [data]
        '''
        sh, sw = utils.calculate_sampling_size(original_shape, self.sampling_factor, max_sampling_factor)
        crop = data[:sh, :sw] # remove padding
        height, width = original_shape
        return cv2.resize(np.uint8(crop), (width, height), interpolation=self.interpolation) # up sampling
    
    def create_block_container(self, shape, max_sfactor, mode = 'non-interleave'):
        '''
        Create a Block Container used for Rearrange blocks
        '''
        ssize = utils.calculate_sampling_size(shape, self.sampling_factor, max_sfactor)
        esize = utils.calculate_padding_size(ssize, self.sampling_factor, mode)
        step      = (1, 1) if mode == 'non-interleave' else (self.sampling_factor[1], self.sampling_factor[0])
        return BlockExtend(step).build(esize)
    