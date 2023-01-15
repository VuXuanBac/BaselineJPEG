import cv2
import numpy as np
from bitarray import bitarray

from bitutils import StateStream
from component import Component
from table import *
import utils

class Frame(object):
    def __init__(self, max_components: int = 3, precision: int = 8) -> None:
        # if mode.lower() != 'baseline':
        #     raise NotImplemented('Support Baseline JPEG only.')
        self.precision              = precision
        self.components             = [Component() for _ in range(max_components)]

        lumaquant, chromaquant      = get_suggest_quant_table()
        lumahuff                    = get_suggest_luma_huffman_table()
        chromahuff                  = get_suggest_chroma_huffman_table()

        self.set_quantization_table([lumaquant, chromaquant])
        self.set_huffman_tables([lumahuff, chromahuff])
        self.set_interpolation('linear')
        self.set_sampling_factor(420)

    def set_huffman_tables(self, tables: tuple[HuffmanTable] | list):
        comp_tables = utils.broadcast(tables, len(self.components))
        for t, comp in zip(comp_tables, self.components):
            comp.huffman_tables = t

    def set_quantization_table(self, table: QuantizationTable | list):
        comp_tables = utils.broadcast(table, len(self.components))
        for t, comp in zip(comp_tables, self.components):
            comp.quantization_table = t

    def set_interpolation(self, name: str | list):
        '''
        Extract sampling strategy by name. See cv2.INTER_
        Support: 'nearest', 'linear', 'cubic', 'area', 'lanczos4', 'nearest-exact', 'linear-exact', 'max'
        '''
        name_list = ['nearest', 'linear', 'cubic', 'area', 'lanczos4', 'nearest-exact', 'linear-exact', 'max']
        interps = []
        if isinstance(name, str):
            name = [name]
        for n in name:
            n = n.lower()
            if n in name_list:
                interps.append(name_list.index(n))
            else:
                interps.append(1)
        comp_interps = utils.broadcast(interps, len(self.components))
        for t, comp in zip(comp_interps, self.components):
            comp.interpolation = t

    def set_quality(self, quality: int | list):
        '''
        Set compression quality. A relative value for the visual result of compress and restore process.
        '''
        comp_quality = utils.broadcast(quality, len(self.components))
        for q, comp in zip(comp_quality, self.components):
            Component.set_quality(comp, q)

    def set_sampling_factor(self, factor: int | str | list):
        '''
        Extract sampling factor from format. (horizontal, vertical)
        Support: 4:A:0 or 4:A:A and 4 is divisible by A.
        '''
        if isinstance(factor, str):
            factor = int(factor[::2])
        if isinstance(factor, int):
            factor = utils.get_sampling_factor(factor)

        comp_factors = utils.broadcast(factor, len(self.components))
        for t, comp in zip(comp_factors, self.components):
            comp.sampling_factor = t

    def _get_orders(self, components: list[Component]) -> list:
        orders = []
        for index, comp in enumerate(components):
            orders.extend([index] * (comp.sampling_factor[0] * comp.sampling_factor[1]))
        return orders

    def _get_max_sampling_factor(self, components: list[Component]) -> tuple:
        r = [0, 0]
        for comp in components:
            for i in range(2):
                r[i] = max(r[i], comp.sampling_factor[i])
        return r

    def encode(self, data: np.ndarray, *, mode: str = 'non-interleave') -> bitarray:
        ### Color space convert ###
        image_type         = 'color' if (len(data.shape) == 3 and data.shape[2] == 3) else 'grey'
        if image_type == 'color':
            ycrcb = cv2.cvtColor(data, cv2.COLOR_BGR2YCrCb)
            component_data  = cv2.split(ycrcb)
            components      = self.components[:3]
        else: # grey
            component_data  = data
            components      = self.components[:1]
            mode            = 'non-interleave'

        max_sfactor = self._get_max_sampling_factor(components)

        ### Preencode (downsampling, expand) -> Level Shift ###
        encode_generators = []
        for component, data in zip(components, component_data):
            pre_data            = component.preencode(data, max_sfactor, mode)
            ### Level Shift ###
            ls = np.array(pre_data, dtype=np.int32) - (1 << (self.precision - 1))
            encode_generators.append(component.encode(ls, mode))

        ### Encode ###
        result = bitarray()
        if mode == 'non-interleave':
            for gen in encode_generators:
                for encoded_data in gen:
                    result.extend(encoded_data)
        else:
            orders = self._get_orders(components)
            try:
                while True:
                    for index in orders:
                        result.extend(next(encode_generators[index]))
            except: # end
                pass

        return result.tobytes()

    def decode(self, data: bytes, image_shape, *, mode: str = 'non-interleave') -> np.ndarray:
        image_type         = 'color' if (len(image_shape) == 3 and image_shape[2] == 3) else 'grey'
        if image_type == 'color':
            components      = self.components[:3]
            component_shape  = image_shape[:2]
        else:
            components      = self.components[:1]
            mode            = 'non-interleave'
            component_shape  = image_shape

        max_sfactor = self._get_max_sampling_factor(components)

        ### Decode ###
        stream = StateStream().feed(data)
        decode_generators = []
        builders = []
        for component in components:
            decode_generators.append(component.decode(stream))
            builders.append(component.create_block_builder(component_shape, max_sfactor, mode))
        
        if mode == 'non-interleave':
            for gen, builder in zip(decode_generators, builders):
                while not builder.end():
                    builder.put_next(next(gen))
        else:
            orders = self._get_orders(components)
            while not builders[0].end():
                for index in orders:
                    builders[index].put_next(next(decode_generators[index]))

        ### Level Shift -> Postdecode (crop, upsampling) -> Merge -> Convert Color ###
        component_data = []
        for index, comp in enumerate(components):
            decoded     = builders[index].get_all()
            ### Level Shift ###
            decoded     = np.clip(decoded + (1 << (self.precision - 1)), 0, (1 << self.precision) - 1)
            component_data.append(comp.postdecode(decoded, max_sfactor, component_shape))
        
        if image_type == 'color':
            merged = cv2.merge(component_data)
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        else:
            return np.array(component_data, dtype=np.uint8)
