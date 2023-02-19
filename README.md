# Simple Baseline JPEG Codecs

References: JPEG Specification part 1.

## Structure

- frame.py: Compressor for one image (frame). Using `set_()` methods for configurations.
- component.py: Compressor for one component (Y, Cb or Cr).
- block.py: Compressor for one block data (8 x 8). Using BlockExtend for rearrange the Blocks.
- bitutils.py: Bit and BitStream utilities.
- utils.py: Some utilities functions.
- table.py: Abstract classes for Quantization Tables and Huffman Tables.
- huffman.py: Huffman Encoder/Decoder created from Huffman Tables.
- reader.py: Read .jpg file format and extract ECS segment (just to compare compression size).
- compare.py: Compare an image and its restore version (PSNR, Luma PSNR, Compression Ratio)
- test.py: Test with our compressor.
- test_cv2.py: Test OpenCV compressor.

## Run Test

```Python
python test.py [path/to/image] [quality]
python test_cv2.py [path/to/image] [quality]
```
