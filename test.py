import compare, sys

from frame import Frame
from utils import load_image

# Input
quality = int(sys.argv[2])
origin = load_image(sys.argv[1])

compressor = Frame()
compressor.set_quality(quality)

# Compress
encoded_data = compressor.encode(origin)

# Restore
restore = compressor.decode(encoded_data, origin.shape)

# Compare
comparer = compare.CompressorAnalysis(origin, restore, len(encoded_data))

print(f'========== Quality {quality} ==========')
print(comparer)
comparer.show(False)