"""
Mold Detection Algorithm for Microgreens
Detects mold in microgreens photos based on:
- Gray color values (darker than seedlings)
- Connections between seedlings > 2mm
- Reduced stalk visibility
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

class MoldDetector:
    def __init__(self, input_dir, output_dir, threshold_config=None):
        """
        Initialize the mold detector.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory for saving results
            threshold_config: Dictionary with detection parameters
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Default thresholds for mold detection
        self.config = {
            'gray_lower': 80,      # Lower gray value threshold
            'gray_upper': 160,     # Upper gray value threshold
            'min_mold_area': 100,  # Minimum pixels for mold region
            'min_connection_length': 10,  # ~2mm in pixels (adjust based on image DPI)
            'stalk_darkness_threshold': 50,  # How dark stalk lines should be
            'blur_kernel': 5,      # For smoothing
            'morphology_kernel': 3,  # For morphological operations
        }
        
        if threshold_config:
            self.config.update(threshold_config)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def remove_stalks_and_leaves(self, image):
        """
        Remove stalks and leaves from the image to reduce noise.
        Stalks and leaves are typically:
        - Dark pixels (low brightness or dark green)
        - Elongated structures (thick and long)
        - Vertically oriented
        
        This function creates a cleaned image by removing these structures.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            cleaned_image: Image with stalks/leaves removed
            mask: Binary mask of removed pixels
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract color components
        h = hsv[:, :, 0]  # Hue
        s = hsv[:, :, 1]  # Saturation
        v = hsv[:, :, 2]  # Value/Brightness
        
        # Dark pixels (stalks are typically very dark)
        dark_mask = gray < 40
        
        # Green pixels (leaves are green) - hue range for green
        # Green is typically between 35-85 in OpenCV HSV (0-180 range)
        green_mask = ((h > 25) & (h < 95)) & (s > 40)
        
        # Dark green leaves (less saturated greens)
        dark_green = ((h > 25) & (h < 95)) & (s > 20) & (v < 150)
        
        # Combine masks for stalk and leaf detection
        stalk_leaf_mask = dark_mask | green_mask | dark_green
        
        # Apply morphological operations to connect nearby pixels (structural elements)
        # Use vertical and horizontal kernels to target elongated structures
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        
        # Close operations to connect broken structures
        stalk_leaf_mask = cv2.morphologyEx(stalk_leaf_mask.astype(np.uint8) * 255, 
                                           cv2.MORPH_CLOSE, vertical_kernel)
        stalk_leaf_mask = cv2.morphologyEx(stalk_leaf_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Dilate to expand the removal area slightly
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stalk_leaf_mask = cv2.dilate(stalk_leaf_mask, dilate_kernel, iterations=2)
        
        # Create cleaned image by inpainting the removed areas
        # Use median filtering to smooth the removed regions
        cleaned_image = image.copy()
        
        # For removed areas, replace with median color of surrounding area
        for channel in range(3):
            channel_data = image[:, :, channel].astype(float)
            median_val = np.median(channel_data[stalk_leaf_mask == 0])
            cleaned_image[stalk_leaf_mask > 0, channel] = median_val
        
        return cleaned_image, stalk_leaf_mask
    
    def remove_stalks_and_leaves(self, image):
        """
        Remove stalks and leaves from the image to reduce noise.
        Stalks and leaves are typically:
        - Dark pixels (low brightness or dark green)
        - Elongated structures (thick and long)
        - Vertically oriented
        
        Args:
            image: Input image (BGR)
            
        Returns:
            cleaned_image: Image with stalks/leaves removed
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract color components
        h = hsv[:, :, 0]  # Hue
        s = hsv[:, :, 1]  # Saturation
        v = hsv[:, :, 2]  # Value/Brightness
        
        # Dark pixels (stalks are typically very dark)
        dark_mask = gray < 40
        
        # Green pixels (leaves are green) - hue range for green
        # Green is typically between 35-85 in OpenCV HSV (0-180 range)
        green_mask = ((h > 25) & (h < 95)) & (s > 40)
        
        # Dark green leaves (less saturated greens)
        dark_green = ((h > 25) & (h < 95)) & (s > 20) & (v < 150)
        
        # Combine masks for stalk and leaf detection
        stalk_leaf_mask = dark_mask | green_mask | dark_green
        
        # Apply morphological operations to connect nearby pixels
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        
        # Close operations to connect broken structures
        stalk_leaf_mask = cv2.morphologyEx(stalk_leaf_mask.astype(np.uint8) * 255, 
                                           cv2.MORPH_CLOSE, vertical_kernel)
        stalk_leaf_mask = cv2.morphologyEx(stalk_leaf_mask, cv2.MORPH_CLOSE, horizontal_kernel)
        
        # Dilate to expand the removal area
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stalk_leaf_mask = cv2.dilate(stalk_leaf_mask, dilate_kernel, iterations=2)
        
        # Inpaint the removed areas with nearby pixels
        cleaned_image = cv2.inpaint(image, stalk_leaf_mask, 3, cv2.INPAINT_TELEA)
        
        return cleaned_image
    
    def detect_gray_areas(self, image):
        """
        Detect gray areas in the image (potential mold).
        
        Args:
            image: Input image (BGR)
            
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
        gray_mask = (saturation < 50) & (gray > self.config['gray_lower']) & (gray < self.config['gray_upper'])
        
        return gray_mask.astype(np.uint8) * 255
    
    def detect_stalk_lines(self, image):
        """
        Detect dark stalk lines in the image.
        Uses edge detection to find vertical/diagonal lines.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            stalk_mask: Binary mask of detected stalk lines
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect dark pixels (stalks are typically darker than background)
        dark_pixels = gray < self.config['stalk_darkness_threshold']
        
        # Use morphological operations to find connected line-like structures
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        stalk_mask = cv2.morphologyEx(dark_pixels.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        
        return stalk_mask
    
    def detect_mold_regions(self, image):
        """
        Detect mold regions using multiple criteria:
        1. Gray color values
        2. Reduced stalk visibility in the area
        3. Connected regions > 2mm
        
        Args:
            image: Input image (BGR)
            
        Returns:
            mold_mask: Binary mask of detected mold regions
            contours: List of contours for mold regions
        """
        # Pre-process: Remove stalks and leaves to reduce noise
        cleaned_image = self.remove_stalks_and_leaves(image)
        
        # Get gray areas (low saturation, mid-range brightness) from cleaned image
        gray_areas = self.detect_gray_areas(cleaned_image)
        
        # Get stalk lines from cleaned image
        stalk_lines = self.detect_stalk_lines(cleaned_image)
        
        # Mold is present where:
        # - We have gray areas
        # - AND stalk visibility is reduced (fewer dark pixels)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        stalk_density = cv2.morphologyEx(stalk_lines, cv2.MORPH_CLOSE, kernel)
        
        # Areas with gray color but low stalk density = likely mold
        reduced_stalk_mask = stalk_density < 50  # Adjust threshold as needed
        
        # Combine criteria
        mold_mask = (gray_areas > 0) & (reduced_stalk_mask)
        mold_mask = mold_mask.astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config['morphology_kernel'], self.config['morphology_kernel']))
        mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_OPEN, kernel)
        mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by minimum area
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.config['min_mold_area']]
        
        return mold_mask, filtered_contours
    
    def draw_mold_circles(self, image, contours):
        """
        Draw red circles around detected mold regions.
        
        Args:
            image: Input image (BGR)
            contours: List of mold contours
            
        Returns:
            result_image: Image with red circles drawn
            mold_found: Boolean indicating if mold was detected
        """
        result_image = image.copy()
        mold_found = len(contours) > 0
        
        # Draw circles around each mold region
        for contour in contours:
            # Get the bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Only draw if radius is significant
            if radius > 5:
                x, y, radius = int(x), int(y), int(radius)
                # Draw red circle
                cv2.circle(result_image, (x, y), radius, (0, 0, 255), 2)
        
        # Add text indicator
        if mold_found:
            cv2.putText(result_image, f"MOLD DETECTED ({len(contours)} regions)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(result_image, "No Mold Detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_image, mold_found
    
    def process_image(self, image_path, save_results=True, verbose=True, save_preprocessing=False):
        """
        Process a single image for mold detection.
        
        Args:
            image_path: Path to the image file
            save_results: Whether to save the result image
            verbose: Whether to print processing information
            save_preprocessing: Whether to save preprocessing visualization
            
        Returns:
            result_image: Image with mold regions circled
            mold_found: Boolean indicating if mold was detected
            stats: Dictionary with detection statistics
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            if verbose:
                print(f"Error: Could not load image {image_path}")
            return None, False, {}
        
        # Optional: Save preprocessing result
        if save_preprocessing:
            cleaned_image, stalk_leaf_mask = self.remove_stalks_and_leaves(image)
            preprocess_output = os.path.join(self.output_dir, f"preprocess_{os.path.basename(image_path)}")
            cv2.imwrite(preprocess_output, cleaned_image)
        
        # Detect mold regions
        mold_mask, contours = self.detect_mold_regions(image)
        
        # Draw circles
        result_image, mold_found = self.draw_mold_circles(image, contours)
        
        # Prepare statistics
        stats = {
            'filename': os.path.basename(image_path),
            'mold_detected': mold_found,
            'mold_regions': len(contours),
            'image_size': image.shape,
        }
        
        if mold_found:
            # Calculate mold coverage percentage
            mold_pixels = np.sum(mold_mask > 0)
            total_pixels = mold_mask.shape[0] * mold_mask.shape[1]
            stats['mold_coverage_percent'] = (mold_pixels / total_pixels) * 100
            stats['region_areas'] = [cv2.contourArea(c) for c in contours]
        else:
            stats['mold_coverage_percent'] = 0
            stats['region_areas'] = []
        
        # Save result if requested
        if save_results:
            output_filename = f"result_{os.path.basename(image_path)}"
            output_path = os.path.join(self.output_dir, output_filename)
            cv2.imwrite(output_path, result_image)
            stats['output_path'] = output_path
            if verbose:
                print(f"Saved: {output_path}")
        
        if verbose:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            print(f"  Mold Detected: {mold_found}")
            print(f"  Regions Found: {len(contours)}")
            print(f"  Mold Coverage: {stats['mold_coverage_percent']:.2f}%")
        
        return result_image, mold_found, stats
    
    def process_directory(self, extensions=None):
        """
        Process all images in the input directory.
        
        Args:
            extensions: List of file extensions to process (default: jpg, png)
            
        Returns:
            results: Dictionary with processing results for all images
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        results = {
            'total_images': 0,
            'mold_detected_count': 0,
            'images': []
        }
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(Path(self.input_dir).glob(f'*{ext}'))
        
        image_files = sorted(image_files)
        results['total_images'] = len(image_files)
        
        print(f"\nProcessing {len(image_files)} images from {self.input_dir}\n")
        print("=" * 60)
        
        for idx, image_path in enumerate(image_files, 1):
            result_image, mold_found, stats = self.process_image(image_path)
            
            if result_image is not None:
                results['images'].append(stats)
                if mold_found:
                    results['mold_detected_count'] += 1
        
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Total images processed: {results['total_images']}")
        print(f"  Images with mold: {results['mold_detected_count']}")
        print(f"  Mold-free images: {results['total_images'] - results['mold_detected_count']}")
        
        return results
    
    def display_comparison(self, image_path):
        """
        Display original and processed image side-by-side.
        
        Args:
            image_path: Path to the image to display
        """
        image = cv2.imread(str(image_path))
        result_image, mold_found, stats = self.process_image(image_path, save_results=False, verbose=False)
        
        # Convert BGR to RGB for matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Display
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(result_rgb)
        axes[1].set_title(f'Mold Detection Result\n(Mold: {stats["mold_detected"]}, Regions: {stats["mold_regions"]})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function."""
    # Configuration
    input_dir = "resources/Photos-Mold"
    output_dir = "resources/Mold_Detection_Results"
    
    # Create detector
    detector = MoldDetector(input_dir, output_dir)
    
    # Process all images
    results = detector.process_directory()
    
    # Save summary report
    report_path = os.path.join(output_dir, "detection_report.txt")
    with open(report_path, 'w') as f:
        f.write("MOLD DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images processed: {results['total_images']}\n")
        f.write(f"Images with mold detected: {results['mold_detected_count']}\n")
        f.write(f"Mold-free images: {results['total_images'] - results['mold_detected_count']}\n")
        f.write("\nDetailed Results:\n")
        f.write("-" * 60 + "\n")
        
        for img_stats in results['images']:
            f.write(f"\nFile: {img_stats['filename']}\n")
            f.write(f"  Mold Detected: {img_stats['mold_detected']}\n")
            f.write(f"  Regions Found: {img_stats['mold_regions']}\n")
            f.write(f"  Mold Coverage: {img_stats['mold_coverage_percent']:.2f}%\n")
            if img_stats['region_areas']:
                f.write(f"  Region Sizes: {[f'{a:.0f}px' for a in img_stats['region_areas']]}\n")
    
    print(f"\nReport saved to: {report_path}")
    
    # Display one image for visual inspection
    print("\nProcessing complete! Results saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
