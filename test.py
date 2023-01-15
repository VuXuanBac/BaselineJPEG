import compare
import sys
from frame import Frame

frame = Frame()

for quality in range(0, 101, 15):
    print(f'========== Quality {quality} ==========')
    frame.set_quality(quality)
    compare.compare(frame, sys.argv[1])