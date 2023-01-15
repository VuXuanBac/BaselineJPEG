from bitarray import bitarray, util

class JpegBitConverter(object):
    '''
        M = (2^n) - 1          m = 2^(n-1)
    JPEG mapping {-M, -M+1,..., -m-1, -m, m, m+1,..., M} to {0, 1, 2,..., M}
        so, if x > 0 <-> f(x) = x
            else     <-> f(x) = x + M = ~x
    Ex: bitarray('0000000') [7 bits] <-> -127
        bitarray('1111111') [7 bits] <-> 127
        bitarray('0110011') [7 bits] <-> -76
        bitarray('1001100') [7 bits] <-> 76
    '''
    def bits2int(bits: bitarray, signed: bool = True) -> int:
        '''
        [bits]: bitarray represent the value, the length is important.
        So, bits2int(bitarray('0010011')) != bits2int(bitarray('10011'))
        '''
        if signed and bits[0] == 0: # negative
            return - util.ba2int(~bits)
        return util.ba2int(bits)

    def int2bits(value: int, signed: bool = True) -> bitarray:
        '''
        :return: bitarray has [precision] elements where [precision] minimum bits to present abs(value).
        '''
        precision = value.bit_length()
        if signed and value < 0:
            return ~ util.int2ba(- value, precision)
        return util.int2ba(value, precision)

class StateStream(object):
    '''
    BitStream that follow number of read bits.
    '''
    def __init__(self) -> None:
        self.data = bitarray()
        self.index = 0
    
    def fromfile(self, path: str) -> 'StateStream':
        '''
        Feed stream with data from binary file
        '''
        with open(path, 'rb') as file:
            self.data.fromfile(file)
        return self
    
    def feed(self, data: bytes) -> 'StateStream':
        self.data.frombytes(data)
        return self

    def __len__(self):
        return len(self.data)

    def end(self) -> bool:
        return self.index >= len(self.data)

    def next_bit(self) -> int:
        '''
        Get 1 bit from stream
        '''
        b = self.data[self.index]
        self.index += 1
        return b
    
    def next_bits(self, length: int) -> bitarray:
        '''
        Get [length] bits from stream
        '''
        f = self.index
        self.index += length
        return self.data[f:self.index]
