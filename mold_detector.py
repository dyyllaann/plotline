"""
Mold Detection Algorithm for Microgreens
Detects mold in microgreens photos based on:
1. Detect stalks using edge detection and contours (vertical lines)
2. Find gray webbing/fuzzy areas between stalks
3. Look for web-like structures connecting stalks
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

class MoldDetector:
    def __init__(self, input_dir, output_dir, threshold_config=None):
        """
        Initialize the mold detector with tunable parameters.
        
        Args:
            threshold_config: Dictionary with detection parameters that can be
                            optimized/trained via grid search, Bayesian optimization, etc.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Paramters for tuning/optimization
        self.config = {
            # Stalk detection
            'canny_low': 50,              # Range: [30-100]
            'canny_high': 150,            # Range: [100-200]
            'min_stalk_length': 30,       # Range: [20-100]
            'stalk_aspect_ratio': 3.0,    # Range: [2.0-5.0]
            
            # Mold webbing detection
            'web_gray_lower': 70,         # Range: [50-100]
            'web_gray_upper': 160,        # Range: [120-200]
            'min_web_area': 200,          # Range: [100-500]
            'max_saturation': 40,         # Range: [20-80]
            
            # Texture analysis
            'texture_kernel': 3,          # Options: [3, 5, 7]
            'blur_kernel': 3,             # Options: [3, 5, 7]
            'morphology_kernel': 3,       # Options: [3, 5, 7, 9]
            'laplacian_threshold': 10,    # Range: [5-30]
        }
        
        if threshold_config:
            self.config.update(threshold_config)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_params(self):
        """Return parameters."""
        return self.config.copy()
    
    def set_params(self, params):
        """Update parameters."""
        self.config.update(params)
    
    def detect_stalks(self, image):
        """
        Detect microgreen stalks using edge detection and contour analysis.
        Stalks are elongated structures (high aspect ratio) in ANY orientation.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            stalk_mask: Binary mask of detected stalks
            stalk_contours: List of stalk contours
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.config['blur_kernel'], self.config['blur_kernel']), 0)
        
        # Edge detection to find stalk boundaries
        edges = cv2.Canny(blurred, self.config['canny_low'], self.config['canny_high'])
        
        # Dilate to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for elongated stalk-like structures (any orientation)
        stalk_contours = []
        stalk_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 20:
                continue
            
            # Use minimum area rectangle to get orientation-independent aspect ratio
            rect = cv2.minAreaRect(contour)
            (center), (width, height), angle = rect
            
            # Ensure width is the smaller dimension
            if width > height:
                width, height = height, width
            
            # Check if it's elongated (thin and long) - orientation independent
            if height > self.config['min_stalk_length'] and width > 0:
                aspect_ratio = height / width
                if aspect_ratio > self.config['stalk_aspect_ratio']:
                    stalk_contours.append(contour)
                    cv2.drawContours(stalk_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        return stalk_mask, stalk_contours
    
    def detect_webbing_between_stalks(self, image, stalk_mask):
        """
        Detect gray, fuzzy mold webbing in areas between stalks.
        This is where we'd normally see soil but instead see gray blobs
        with fine web-like structures.
        
        Args:
            image: Input image (BGR)
            stalk_mask: Binary mask of detected stalks
            
        Returns:
            web_mask: Binary mask of detected mold webbing
            web_contours: List of webbing contours
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Create mask for gray areas (low saturation, medium value)
        gray_mask = cv2.inRange(s, 0, self.config['max_saturation'])
        value_mask = cv2.inRange(v, self.config['web_gray_lower'], self.config['web_gray_upper'])
        
        # Combine masks: gray AND medium brightness
        web_mask = cv2.bitwise_and(gray_mask, value_mask)
        
        # Remove stalk areas - we only care about what's BETWEEN stalks
        web_mask = cv2.bitwise_and(web_mask, cv2.bitwise_not(stalk_mask))
        
        # Detect fine web-like texture using Laplacian for texture detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=self.config['texture_kernel'])
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Threshold to get web-like fine lines
        _, texture_mask = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
        
        # Combine gray blob detection with fine texture detection
        # Mold = gray blob + fine web texture
        web_mask = cv2.bitwise_and(web_mask, texture_mask)
        
        # Morphological operations to clean up and connect regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.config['morphology_kernel'], 
                                           self.config['morphology_kernel']))
        web_mask = cv2.morphologyEx(web_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        web_mask = cv2.morphologyEx(web_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours of webbing regions
        contours, _ = cv2.findContours(web_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area
        web_contours = [c for c in contours if cv2.contourArea(c) >= self.config['min_web_area']]
        
        return web_mask, web_contours
    
    def detect_mold_regions(self, image):
        """
        Main detection pipeline:
        1. Detect stalks (multi-directional contours)
        2. Find gray webbing between stalks
        
        Args:
            image: Input image (BGR)
            
        Returns:
            mold_mask: Binary mask of detected mold regions
            contours: List of contours for mold regions
            debug_info: Dictionary with intermediate results for visualization
        """
        # Step 1: Detect stalks
        stalk_mask, stalk_contours = self.detect_stalks(image)
        
        # Step 2: Detect webbing between stalks
        web_mask, web_contours = self.detect_webbing_between_stalks(image, stalk_mask)
        
        debug_info = {
            'stalk_mask': stalk_mask,
            'stalk_contours': stalk_contours,
            'web_mask': web_mask
        }
        
        return web_mask, web_contours, debug_info
    
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
        
        for contour in contours:
            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw red circle
            cv2.circle(result_image, center, radius, (0, 0, 255), 2)
            
            # Add label
            cv2.putText(result_image, "MOLD", 
                       (center[0] - 20, center[1] - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        mold_found = len(contours) > 0
        return result_image, mold_found
    
    def process_image(self, image_path, save_results=True, verbose=True, save_debug=False):
        """
        Process a single image for mold detection.
        
        Args:
            image_path: Path to the image file
            save_results: Whether to save the result image
            verbose: Whether to print processing information
            save_debug: Whether to save debug visualizations
            
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
        
        # Detect mold regions
        mold_mask, contours, debug_info = self.detect_mold_regions(image)
        
        # Save debug visualization if requested
        if save_debug:
            debug_output = os.path.join(self.output_dir, f"debug_{os.path.basename(image_path)}")
            debug_vis = np.hstack([
                cv2.cvtColor(debug_info['stalk_mask'], cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(debug_info['web_mask'], cv2.COLOR_GRAY2BGR)
            ])
            cv2.imwrite(debug_output, debug_vis)
        
        # Draw circles
        result_image, mold_found = self.draw_mold_circles(image, contours)
        
        # Prepare statistics
        stats = {
            'filename': os.path.basename(image_path),
            'mold_detected': mold_found,
            'mold_regions': len(contours),
            'stalks_detected': len(debug_info['stalk_contours']),
            'image_size': image.shape,
        }
        
        if mold_found:
            mold_pixels = np.sum(mold_mask > 0)
            total_pixels = mold_mask.shape[0] * mold_mask.shape[1]
            stats['mold_coverage_percent'] = (mold_pixels / total_pixels) * 100
            stats['region_areas'] = [cv2.contourArea(c) for c in contours]
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
            print(f"  Stalks Detected: {stats['stalks_detected']}")
            print(f"  Mold Detected: {mold_found}")
            print(f"  Mold Regions: {len(contours)}")
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
