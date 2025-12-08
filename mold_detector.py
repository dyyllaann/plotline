
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

# Import utility functions from mold_utils
from mold_utils import (
    remove_stalks_and_leaves,
    detect_gray_areas,
    filter_contours,
    draw_results
)

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
            'min_mold_area': 200,  # Minimum pixels for mold region (more sensitive)
            'min_connection_length': 10,  # ~2mm in pixels (adjust based on image DPI)
            'stalk_darkness_threshold': 50,  # How dark stalk lines should be
            'blur_kernel': 2,      # For smoothing
            'morphology_kernel': 3,  # For morphological operations
            'min_circularity': 0.15,  # Lower threshold - more permissive (was 0.3)
        }
        
        if threshold_config:
            self.config.update(threshold_config)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def detect_mold_regions(self, image):
        """
        Detect mold regions using multiple criteria:
        1. Remove bright stalks/leaves (preprocessing)
        2. Detect gray color values (low saturation, mid-range brightness)
        3. Apply morphological operations to connect regions
        4. Filter by size and shape
        
        Args:
            image: Input image (BGR)
            
        Returns:
            mold_mask: Binary mask of detected mold regions
            contours: List of contours for mold regions
        """
        # Pre-process: Remove stalks and leaves to reduce noise (using mold_utils)
        cleaned_image, _ = remove_stalks_and_leaves(image)
        
        # Get gray areas (low saturation, mid-range brightness) from cleaned image
        # This is where mold appears - gray, fuzzy, web-like
        gray_areas = detect_gray_areas(
            cleaned_image,
            gray_lower=self.config['gray_lower'],
            gray_upper=self.config['gray_upper']
        )
        
        # Use gray areas as our mold mask
        mold_mask = gray_areas
        
        # Apply morphological operations with larger kernel to connect regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.config['morphology_kernel'], 
                                           self.config['morphology_kernel']))
        mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_OPEN, kernel)
        mold_mask = cv2.morphologyEx(mold_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Apply multiple filters to be more selective (using mold_utils)
        filtered_contours = filter_contours(
            contours,
            min_area=self.config['min_mold_area'],
            min_circularity=self.config['min_circularity']
        )
        
        return mold_mask, filtered_contours
    
    def draw_mold_circles(self, image, contours):
        """
        Draw red circles around ALL detected mold regions (uses mold_utils).
        
        Args:
            image: Input image (BGR)
            contours: List of mold contours
            
        Returns:
            result_image: Image with red circles drawn
            mold_found: Boolean indicating if mold was detected
        """
        # Use the draw_results function from mold_utils
        result_image = draw_results(image, contours)
        mold_found = len(contours) > 0
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
            
            # Add region centers
            stats['region_centers'] = []
            for contour in contours:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                stats['region_centers'].append((int(x), int(y)))
        else:
            stats['mold_coverage_percent'] = 0
            stats['region_areas'] = []
            stats['region_centers'] = []
        
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
    output_dir = "resources/Mold_Detection_Results-Dylan"
    
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
                f.write(f"  Region Sizes: {[f'{a:.0f}px²' for a in img_stats['region_areas']]}\n")
                f.write(f"  Region Centers (x,y): {img_stats['region_centers']}\n")
    
    print(f"\nReport saved to: {report_path}")
    
    # Display one image for visual inspection
    print("\nProcessing complete! Results saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
