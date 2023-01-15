import cv2
import numpy as np
from utils import load_image

class CompressorStatistic(object):
    '''
    Compare between original image with its compressed image.
    '''

    def __init__(self, origin, restore, compressed_size) -> None:
        self.origin             = origin
        self.restore            = restore
        # self.quality            = quality
        self.compressed_size    = compressed_size
        self.psnr               = cv2.PSNR(origin, restore)

        luma = lambda image: 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]

        self.luma_psnr          = cv2.PSNR(luma(origin), luma(restore))
        self.difference_image   = np.abs(origin - restore)

    def __str__(self) -> str:
        oshape = self.origin.shape
        size = f'Image Dimension\t\t: {oshape[1]} x {oshape[0]}'
        # quality = f'Quality\t\t\t: {self.quality}'
        psnr = f'PSNR\t\t\t: {self.psnr:.2f} dB'
        luma_psnr = f'Luma PSNR\t\t: {self.luma_psnr:.2f} dB'
        osize = oshape[0] * oshape[1] * oshape[2]
        ratio = osize // self.compressed_size
        compress_ratio = f'Compress Ratio\t\t: {ratio}:1 [{osize} B -> {self.compressed_size} B]'
        return f'{size}\n{psnr}\n{luma_psnr}\n{compress_ratio}'

    def show(self):
        '''
        Show origin image, compressed image and difference image on separate windows.
        '''
        # y, cr, cb = cv2.split(self.difference_image)
        titles = ['Origin', 'Compressed', 'Difference']#, 'Diff-Y', 'Diff-Cr', 'Diff-Cb']
        images = [self.origin, self.restore, self.difference_image]#, y, cr, cb]
        for img, til in zip(images, titles):
            cv2.imshow(til, img)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def compare(compressor, path: str, encode_mode = 'non-interleave') -> None:
    origin = load_image(path)
    encoded_data = compressor.encode(origin, mode = encode_mode)
    restore = compressor.decode(encoded_data, origin.shape, mode = encode_mode)
    comparer = CompressorStatistic(origin, restore, len(encoded_data))
    print(comparer)
    comparer.show()
