# Simple Baseline JPEG Codecs

References: JPEG Specification part 1.

## Structure

- frame.py: Compressor for one image (frame), Start with Frame object and set its attribute with `set_` methods.
- component.py: Compressor for one component (Y, Cb or Cr).
- block.py: Block with DCT, Quantization and BlockExtend for arrange the Blocks.
- bitutils.py: Bit and BitStream Utilities.
- utils.py: Some utilities functions.
- table.py: Abstract classes for Quantization Table and Huffman Tables.
- huffman.py: Encoder/Decoder from Huffman Tables.
- compare.py: Compare between an image and its restore version.
- test.py: Test compressor with image in different quality.

## Run Test

```Python
python test.py [path/to/image]
```
