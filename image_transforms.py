import numpy as np
from PIL import Image, ImageOps

# square padding using pillow: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
class SquarePad(object):
    def __init__(self, single_channel=False):
    
        self.pad = (int(0.333 * 256), int(0.333 * 256), int(0.333 * 256))

    def __call__(self, img):
        '''
        Parameters:
        -----------
        img: Image
            image to be padded
        
        Returns:
        image_padded: Image
            image padded with fill value
        '''
        w, h = img.size
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=self.pad)
        return img_padded

        
class DepthNormalize(object):
    '''
    Converts depth image to standard normal image
    '''

    def __call__(self, depth_img):
        '''
        Parameters:
        ----------
        depth_img: Tensor
            depth image tensor
        
        Returns:
        -------
        depth_map: Tensor
            normalised depth map
        '''
       
        depth_img = depth_img.float()
        _, height, width = depth_img.size()
        zero_avoid = 1.0 / np.sqrt(height * width)
        depth_map = depth_img - depth_img.mean()
        depth_map /= np.maximum(depth_map.std(), zero_avoid)
        return depth_map

class RepeatTensor(object):
    '''
    Converts depth image into 3 channels
    '''

    def __call__(self, img):
        '''
        Parameters:
        ----------
        img: Tensor
            image tensor
        
        Returns:
        -------
        img: Tensor
           image repeated across 3 channels
        '''
        return img.repeat(3, 1, 1)
