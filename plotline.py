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

# Distance from c1 center to c2 center
def circle_distance(c1, c2):
    x1, y1 = c1.center
    x2, y2 = c2.center
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Do c1 and c2 overlap
def circles_overlap(c1, c2):
    dist = circle_distance(c1, c2)
    return dist <= (c1.radius + c2.radius)

# Is c_test inside c_gt
def circle_inside(c_test, c_gt):
    dist = circle_distance(c_test, c_gt)
    return dist + c_test.radius <= c_gt.radius
