"""
Standalone utility functions for mold detection.
These can be called individually in notebooks for testing and tuning.
"""

import cv2
import numpy as np

def remove_stalks_and_leaves(image):
    """
    Remove bright stalks and green leaves from the image.
    Composed of detect_stalk_lines() and detect_leaf_areas().
    Mold is typically gray (low saturation, mid-range brightness).
    
    Args:
        image: Input image (BGR)
        
    Returns:
        cleaned_image: Image with stalks/leaves masked
        stalk_leaf_mask: Binary mask of removed areas (for debugging)
    """
    # Detect stalks and leaves using component functions
    stalk_mask = detect_stalk_lines(image)
    leaf_mask = detect_leaf_areas(image)
    
    # Combine masks
    stalk_leaf_mask = cv2.bitwise_or(stalk_mask, leaf_mask)
    
    # Morphological operations to connect stalk segments
    # Use larger kernel to connect broken stalk lines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stalk_leaf_mask = cv2.morphologyEx(stalk_leaf_mask, cv2.MORPH_CLOSE, kernel)
    
    # Dilate to widen stalks and ensure complete removal
    stalk_leaf_mask = cv2.dilate(stalk_leaf_mask, kernel, iterations=4)
    
    # Inpaint the removed areas
    cleaned_image = cv2.inpaint(image, stalk_leaf_mask, 3, cv2.INPAINT_TELEA)
    
    return cleaned_image, stalk_leaf_mask


def visualize_masking(image, mask):
    """
    Show what remains in the image after masking.
    
    Args:
        image: Original image (BGR)
        mask: Binary mask (255 = remove, 0 = keep)
        
    Returns:
        masked_image: Image with masked areas set to black
        inverse_masked: Image showing only masked areas
    """
    # Convert mask to 3 channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Image with masked areas removed (set to black)
    masked_image = image.copy()
    masked_image[mask > 0] = 0
    
    # Image showing only masked areas (inverse)
    inverse_masked = np.zeros_like(image)
    inverse_masked[mask > 0] = image[mask > 0]
    
    return masked_image, inverse_masked


def detect_gray_areas(image, gray_lower=80, gray_upper=160):
    """
    Detect gray areas in the image (potential mold).
    
    Args:
        image: Input image (BGR)
        gray_lower: Lower threshold for gray value
        gray_upper: Upper threshold for gray value
        
    Returns:
        mask: Binary mask of gray areas
    """
    # Convert to HSV and grayscale for analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract saturation channel - gray areas have low saturation
    saturation = hsv[:, :, 1]
    
    # Create mask for low saturation (grayish) areas
    # Gray areas have low saturation and mid-range brightness
    gray_mask = (saturation < 50) & (gray > gray_lower) & (gray < gray_upper)
    
    return gray_mask.astype(np.uint8) * 255


def detect_bright_areas(image):
    """
    Detect bright/light colored areas (stalks by color).
    
    Args:
        image: Input image (BGR)
        
    Returns:
        bright_mask: Binary mask of bright areas
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract color components
    h = hsv[:, :, 0]  # Hue
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value/Brightness
    
    # Bright white/light pixels (seedling stalks are very light)
    bright_mask = gray > 160  # Catch lighter stalks
    
    # Light pixels with low saturation (whitish/pale stalks)
    # These are bright but not very colorful - NOT gray mold
    pale_mask = (v > 150) & (s < 100)  # Bright but not saturated = stalks
    
    # Yellow-green tinted stalks (common in seedlings)
    yellow_green_mask = ((h > 20) & (h < 50)) & (v > 140) & (s < 120)
    
    # Combine all bright area detection criteria
    combined_mask = bright_mask | pale_mask | yellow_green_mask
    
    return combined_mask.astype(np.uint8) * 255


def detect_stalk_lines(image, min_line_length=20, thickness_range=(3, 15)):
    """
    Detect elongated line structures (stalks by shape).
    Stalks are characterized by long, curved, thick lines.
    
    Args:
        image: Input image (BGR)
        min_line_length: Minimum length for a line to be considered a stalk
        thickness_range: (min, max) thickness in pixels
        
    Returns:
        line_mask: Binary mask of detected line structures
    """
    # Start with bright areas as candidates
    bright_mask = detect_bright_areas(image)
    
    # Find contours in the bright mask
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by shape - look for elongated structures
    line_mask = np.zeros_like(bright_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # Skip very small regions
            continue
        
        # Fit a bounding rectangle
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        
        # Ensure width/height are positive
        if width == 0 or height == 0:
            continue
        
        # Calculate aspect ratio (length / width)
        length = max(width, height)
        thickness = min(width, height)
        aspect_ratio = length / thickness if thickness > 0 else 0
        
        # Check if it's line-like: long relative to width
        is_elongated = aspect_ratio > 3  # At least 3:1 ratio
        is_long_enough = length > min_line_length
        is_right_thickness = thickness_range[0] <= thickness <= thickness_range[1]
        
        if is_elongated and is_long_enough and is_right_thickness:
            # Draw filled contour on the line mask
            cv2.drawContours(line_mask, [contour], -1, 255, -1)
    
    return line_mask


def detect_leaf_areas(image):
    """
    Detect green leaf areas in the image.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        leaf_mask: Binary mask of detected leaf areas
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract color components
    h = hsv[:, :, 0]  # Hue
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value/Brightness
    
    # Bright green pixels (leaves) - more selective
    green_mask = ((h > 30) & (h < 90)) & (s > 60) & (v > 80)
    
    return green_mask.astype(np.uint8) * 255


def detect_mold_mask(image, gray_lower=80, gray_upper=160, morphology_kernel=7):
    """
    Create a binary mask of potential mold regions.
    Simplified: Remove stalks, then detect gray areas.
    
    Args:
        image: Input image (BGR)
        gray_lower: Lower threshold for gray detection
        gray_upper: Upper threshold for gray detection
        morphology_kernel: Kernel size for morphological operations
        
    Returns:
        mold_mask: Binary mask of mold regions
        cleaned_image: Preprocessed image (for debugging)
    """
    # Pre-process: Remove stalks and leaves
    cleaned_image, _ = remove_stalks_and_leaves(image)
    
    # Detect gray areas (mold is gray, fuzzy, web-like)
    gray_areas = detect_gray_areas(cleaned_image, gray_lower, gray_upper)
    
    # Use gray areas as our mold mask
    mold_mask = gray_areas
    
    # Apply morphological operations to connect regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                      (morphology_kernel, morphology_kernel))
    mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_OPEN, kernel)
    mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_CLOSE, kernel)
    
    return mold_mask, cleaned_image


def filter_contours(contours, min_area=200, min_circularity=0.15):
    """
    Filter contours by area and shape.
    
    Args:
        contours: List of contours to filter
        min_area: Minimum contour area
        min_circularity: Minimum circularity (0-1)
        
    Returns:
        filtered_contours: List of filtered contours
    """
    filtered = []
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Filter 1: Minimum area
        if area < min_area:
            continue
        
        # Filter 2: Circularity
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < min_circularity:
                continue
        
        # Filter 3: Area ratio
        (x, y), radius = cv2.minEnclosingCircle(c)
        circle_area = np.pi * radius ** 2
        area_ratio = area / circle_area if circle_area > 0 else 0
        if area_ratio < 0.25:
            continue
        
        filtered.append(c)
    
    return filtered


def draw_results(image, contours):
    """
    Draw circles around detected regions.
    
    Args:
        image: Input image (BGR)
        contours: List of contours to draw
        
    Returns:
        result_image: Image with circles drawn
    """
    result_image = image.copy()
    
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        x, y, radius = int(x), int(y), max(int(radius), 5)
        
        # Draw red circle
        cv2.circle(result_image, (x, y), radius, (0, 0, 255), 2)
        # Draw center marker
        cv2.circle(result_image, (x, y), 3, (255, 0, 0), -1)
    
    # Add text
    if len(contours) > 0:
        cv2.putText(result_image, f"MOLD DETECTED ({len(contours)} regions)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(result_image, "No Mold Detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_image


def full_pipeline(image, **kwargs):
    """
    Run the full mold detection pipeline.
    
    Args:
        image: Input image (BGR)
        **kwargs: Optional parameters (gray_lower, gray_upper, morphology_kernel, 
                  min_area, min_circularity)
        
    Returns:
        result_image: Image with detections drawn
        contours: List of detected contours
        stats: Dictionary with detection statistics
    """
    # Get mask
    mold_mask, cleaned_image = detect_mold_mask(
        image, 
        gray_lower=kwargs.get('gray_lower', 80),
        gray_upper=kwargs.get('gray_upper', 160),
        morphology_kernel=kwargs.get('morphology_kernel', 7)
    )
    
    # Find contours
    contours, _ = cv2.findContours(mold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    filtered_contours = filter_contours(
        contours,
        min_area=kwargs.get('min_area', 200),
        min_circularity=kwargs.get('min_circularity', 0.15)
    )
    
    # Draw results
    result_image = draw_results(image, filtered_contours)
    
    # Calculate stats
    stats = {
        'mold_detected': len(filtered_contours) > 0,
        'mold_regions': len(filtered_contours),
        'region_areas': [cv2.contourArea(c) for c in filtered_contours],
        'region_centers': [tuple(map(int, cv2.minEnclosingCircle(c)[0])) for c in filtered_contours]
    }
    
    if stats['mold_detected']:
        mold_pixels = np.sum(mold_mask > 0)
        total_pixels = mold_mask.shape[0] * mold_mask.shape[1]
        stats['mold_coverage_percent'] = (mold_pixels / total_pixels) * 100
    else:
        stats['mold_coverage_percent'] = 0
    
    return result_image, filtered_contours, stats
