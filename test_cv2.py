import compare, reader, cv2, sys

# Input
quality = int(sys.argv[2])
origin = cv2.imread(sys.argv[1])

# Compress
cv2.imwrite('test-cv2.jpg', origin, (cv2.IMWRITE_JPEG_QUALITY, quality))

# Restore
restore = cv2.imread('test-cv2.jpg')
encoded_len = reader.get_ecs_length('test-cv2.jpg')

# Compare
comparer = compare.CompressorStatistic(origin, restore, encoded_len)

print(f'========== Quality {quality} ==========')
print(comparer)
# comparer.show(False)