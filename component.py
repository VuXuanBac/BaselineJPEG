import cv2
import numpy as np

import utils
from bitutils import StateStream
from block import BlockExtend
from huffman import HuffmanEncoder, HuffmanDecoder

class Component(object):
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
        Perform Downsampling and Padding (to proper size) the source data.
        '''
        sh, sw = utils.calculate_sampling_size(data.shape[:2], self.sampling_factor, max_sfactor)
        sampling = cv2.resize(data, (sw, sh), interpolation=self.interpolation) # down sampling
        
        eh, ew = utils.calculate_padding_size((sh, sw), self.sampling_factor, mode)
        return cv2.copyMakeBorder(sampling, 0, eh - sh, 0, ew - sw, cv2.BORDER_REPLICATE) # expand to divisible
        # return utils.downsampling_expand(data, self.sampling_factor, max_sfactor, divisors, self.interpolation)
        
        # sampling_size, padding_amount = self.calculate_preencode_size(data.shape, max_sfactor, mode)
        # sampling = cv2.resize(data, dsize=(sampling_size[1], sampling_size[0]), interpolation=self.interpolation) # Expect data.dtype = np.uint8
        # return cv2.copyMakeBorder()
        # sampling_size, padding_amount = self.calculate_preencode_size(data.shape, max_sfactor, mode)
        # # Expand
        # expanded = cv2.copyMakeBorder(data, 0, padding_amount[0], 0, padding_amount[1], cv2.BORDER_REPLICATE)
        # # Down Sampling
        # return cv2.resize(expanded, dsize=(sampling_size[1], sampling_size[0]), interpolation=self.interpolation) # Expect data.dtype = np.uint8

    def encode(self, data: np.ndarray, mode: str = 'non-interleave'):
        pred            = 0
        huffencoder     = HuffmanEncoder(*self.huffman_tables)

        step      = (1, 1) if mode == 'non-interleave' else (self.sampling_factor[1], self.sampling_factor[0])
        blockextend     = BlockExtend(step).feed(data)

        quanttable = self.quantization_table.scale(utils.compute_scale_factor(self.quality))
        while not blockextend.end():
            block = blockextend.get_next().fdct().quantize(quanttable)
            yield huffencoder.encode(block, pred = pred)
            pred    = block.get_dc()

    def postdecode(self, data: np.ndarray, max_sampling_factor, original_shape) -> np.ndarray:
        sh, sw = utils.calculate_sampling_size(original_shape, self.sampling_factor, max_sampling_factor)
        crop = data[:sh, :sw] # remove padding
        height, width = original_shape
        return cv2.resize(np.uint8(crop), (width, height), interpolation=self.interpolation) # up sampling

        return utils.crop_upsampling(data, original_shape, self.sampling_factor, max_sampling_factor, self.interpolation)
        # hf, vf          = self.sampling_factor
        # mhf, mvf        = max_sampling_factor
        # sh, sw          = data.shape
        # height, width   = original_shape
        
        # eh, ew          = (sh * mvf) // vf, (sw * mhf) // hf
        # # Up Sampling
        # upsampled = cv2.resize(np.uint8(data), dsize=(ew, eh), interpolation=self.interpolation) # Expect data.dtype = np.uint8
        # print(upsampled.shape, upsampled[:height, :width].shape)
        # # Remove Padding
        # return np.array(upsampled[:height, :width], dtype=np.uint8)

    def decode(self, stream: StateStream, mode = 'non-interleave'):
        '''
        Perform Dequantization -> IDCT -> Level Shift on source data, block by block
        :param: [prev_compress_shape] The output shape, in fact, if this process follow by a encode() process.
                this argument is the output shape of preencode() process.
        '''
        pred            = 0
        huffdecoder     = HuffmanDecoder(*self.huffman_tables)

        quanttable = self.quantization_table.scale(utils.compute_scale_factor(self.quality))
        while not stream.end():
            block   = huffdecoder.decode(stream, pred = pred)
            pred    = block.get_dc()
            yield block.dequantize(quanttable).idct().get_copy()

    def create_block_builder(self, shape, max_sfactor, mode = 'non-interleave'):
        ssize = utils.calculate_sampling_size(shape, self.sampling_factor, max_sfactor)
        esize = utils.calculate_padding_size(ssize, self.sampling_factor, mode)
        step      = (1, 1) if mode == 'non-interleave' else (self.sampling_factor[1], self.sampling_factor[0])
        return BlockExtend(step).build(esize)
    