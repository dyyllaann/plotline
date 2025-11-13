import numpy as np
from skimage import filters
from skimage.util.shape import view_as_blocks

def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten 3D matrix of histograms into a 1D feature vector.
    4. Normalize flattened histogram feature vector by L2 norm
       Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    hists = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(rows):
      for j in range(cols):
          cell_magnitude = G_cells[i, j]
          cell_angle = theta_cells[i, j]

          for r in range(cell_magnitude.shape[0]):
            for c in range(cell_magnitude.shape[1]):
                angle = cell_angle[r, c]
                magnitude = cell_magnitude[r, c]

                bin_idx = int(angle // degrees_per_bin) % n_bins
                hists[i, j, bin_idx] += magnitude

    block = hists.flatten()

    norm = np.linalg.norm(block)
    if norm == 0:
        norm = 1
    block = block / norm

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### YOUR CODE HERE

    return block