import numpy as np
from skimage import filters
# from skimage.feature import corner_peaks
# from skimage.util.shape import view_as_blocks
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cdist
from utils import pad

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

def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []

    # Describe the patch by normalizing the image values into a standard
    # normal distribution (having mean of 0 and standard deviation of 1)
    # and then flattening into a 1D array.

    # Find mean and standard deviation of the patch
    mean = np.mean(patch)
    sigma = np.std(patch)

    # If a denominator is zero, divide by 1 instead.
    if sigma == 0:
        normalized = (patch - mean) / 1
    else:
        normalized = (patch - mean) / sigma

    # flatten into a 1D array
    feature = normalized.flatten()

    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be STRICTLY SMALLER
    than the threshold (NOT equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    """Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be STRICTLY SMALLER
    than the threshold (NOT equal to). Return the matches as pairs of vector indices."""

    for i in range(M):
        # Sort distances for this descriptor to find closest and second-closest
        sorted = np.sort(dists[i])
        d1 = sorted[0] # Closest distance
        d2 = sorted[1] # Second closest distance

        # Find index of the closest match
        closest_index = np.argmin(dists[i])

        # Check ratio of the distances are SMALLER than the threshold
        ratio = d1 / d2
        if ratio < threshold:
            matches.append([i, closest_index])

    # Return the matches as pairs of vector indices
    matches = np.asarray(matches)

    return matches

def fit_affine_matrix(p1, p2):
    """
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints

    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = np.pad(p1, ((0, 0), (0, 1)), mode='constant', constant_values=1)
    p2 = np.pad(p2, ((0, 0), (0, 1)), mode='constant', constant_values=1)

    H = np.linalg.lstsq(p2, p1, rcond=None)[0]

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H

def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers (use >, i.e. break ties by whichever set is seen first)
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0

    # RANSAC iteration start

    # Note: while there're many ways to do random sampling, we use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the auto-grader.
    # Sample with this code: 
    for i in range(n_iters):
        # 1. Select random set of matches
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])
    
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 2. Compute affine transformation matrix
        H_sample = np.linalg.lstsq(sample2, sample1, rcond=None)[0]
        
        # 3. Compute inliers via Euclidean distance
        distances = np.sqrt(np.sum((np.dot(matched2, H_sample) - matched1) ** 2, axis=1))
        
        # 4. Keep the largest set of inliers
        inliers = distances < threshold

        if np.sum(inliers) > n_inliers:
            n_inliers = np.sum(inliers)
            max_inliers = inliers.copy()

    # 5. Re-compute least-squares estimate on all inliers
    H = np.linalg.lstsq(matched2[max_inliers], matched1[max_inliers], rcond=None)[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return H, orig_matches[max_inliers]