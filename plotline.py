import numpy as np
from skimage import filters
# from skimage.feature import corner_peaks
# from skimage.util.shape import view_as_blocks
# from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

def harris_corners(img, window_size, k):
    """
    Compute Harris corner response map.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    # Convolve using scipy.ndimage.filters.convolve
    Mxx = convolve(dx * dx, window, mode='constant', cval=0)
    Myy = convolve(dy * dy, window, mode='constant', cval=0)
    Mxy = convolve(dx * dy, window, mode='constant', cval=0)

    # Compute Harris corner response -- R = det(M) - k * (trace(M))^2
    det_M = Mxx * Myy - Mxy * Mxy
    trace_M = Mxx + Myy
    response = det_M - k * (trace_M ** 2)

    return response