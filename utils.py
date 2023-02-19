import cv2
import numpy as np
import bitarray

ZigZagOrder = [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
def tolist_zigzag(data: np.ndarray, item_type_converter = int) -> list:
    '''
    Convert 2D data (8 x 8) to 1D data (64) using zigzag order.
    :param: [item_type_converter] Element type converter for the result.
    '''
    size = 8
    result = [0] * (size * size)
    for r in range(size):
        for c in range(size):
            result[ZigZagOrder[r][c]] = item_type_converter(data[r][c])
    return result

def fromlist_zigzag(data: list, item_type = np.int32) -> np.ndarray:
    '''
    Convert 1D data (64) to 2D data (8 x 8) using zigzag order.
    :param: [item_type] Element type for the result.
    '''
    size    = 8
    result  = [[0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            result[r][c] = data[ZigZagOrder[r][c]]
    return np.array(result, dtype=item_type)
    
def round_up(value: int, divisor: int) -> int:
    '''
    Round [value] to nearest larger number that divisible by [divisor]
    '''
    return value if value % divisor == 0 else (value // divisor + 1) * divisor

def load_image(path: str):
    return cv2.imread(path)

def save_encoded_image(path: str, data: bitarray):
    with open(path, 'wb') as file:
        data.tofile(file)

def calculate_sampling_size(source_shape, sfactor, max_sfactor):
    return (source_shape[0] * sfactor[1] + max_sfactor[1] - 1) // max_sfactor[1], \
        (source_shape[1] * sfactor[0] + max_sfactor[0] - 1) // max_sfactor[0]

def calculate_padding_size(sampling_size, sfactor, mode = 'non-interleave'):
    if mode == 'non-interleave':
        return round_up(sampling_size[0], 8), round_up(sampling_size[1], 8)
    else:
        return round_up(sampling_size[0], 8 * sfactor[0]), round_up(sampling_size[1], 8 * sfactor[1])

def get_sampling_factor(factor: int) -> tuple:
    if   factor == 444:     lf = (1, 1)
    elif factor == 440:     lf = (1, 2)
    elif factor == 420:     lf = (2, 2)
    elif factor == 422:     lf = (2, 1)
    elif factor == 410:     lf = (4, 2)
    elif factor == 411:     lf = (4, 1)
    else:                   lf = (2, 2)
    return [lf, (1, 1)]
    
def broadcast(array: list, length: int) -> list:
    '''
    Broadcast list to new list with length [length].
    '''
    if isinstance(array, list):
        if len(array) > length:
            return array[:length]
        else:
            for _ in range(length - len(array)):
                array.append(array[-1])
        return array
    else:
        return [array] * length

def compute_scale_factor(quality: int) -> float:
    '''
    From [quality], calculate the factor for scaling the quantization table coefs.
    '''
    if quality not in range(1, 100):
        return None
    if 1 <= quality <= 50:
        return 50 / quality
    else:
        return 2.0 - 0.02 * quality