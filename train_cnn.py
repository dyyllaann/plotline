"""
Main script to run mold detection with optional CNN verification.
"""

import os
import argparse
from pathlib import Path
from mold_detector import MoldDetector
from mold_cnn import CNNClassifier
import cv2

class MoldDetectionPipeline:
    """Orchestrates traditional detector + CNN classifier."""
    
    def __init__(self, input_dir, output_dir, use_cnn=False, cnn_model_path=None, config=None):
        """
        Initialize detection pipeline.
        
        Args:
            input_dir: Input images directory
            output_dir: Output results directory
            use_cnn: Whether to use CNN for verification
            cnn_model_path: Path to trained CNN model
            config: Configuration dictionary
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_cnn = use_cnn
        
        # Create output subdirectories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/traditional", exist_ok=True)
        if use_cnn:
            os.makedirs(f"{output_dir}/cnn_verified", exist_ok=True)
        
        # Initialize traditional detector
        self.detector = MoldDetector(input_dir, f"{output_dir}/traditional", config)
        
        # Initialize CNN classifier if requested
        self.cnn_classifier = None
        if use_cnn and cnn_model_path:
            print(f"Loading CNN model from {cnn_model_path}...")
            self.cnn_classifier = CNNClassifier(cnn_model_path)
            print("CNN model loaded successfully!")
    
    def process_with_cnn(self, image_path, top_n=5):
        """
        Process image with traditional detector + CNN verification.
        
        Args:
            image_path: Path to input image
            top_n: Number of top candidates to verify
            
        Returns:
            results: Dictionary with detection results
        """
        # Load image
        image = cv2.imread(str(image_path))
        
        # Get candidates from traditional detector
        mold_mask, ranked_contours = self.detector.detect_mold_regions(image, top_n=top_n*2)
        
        print(f"  Traditional detector found {len(ranked_contours)} candidates")
        
        # Verify with CNN
        verified_contours = []
        for contour, trad_confidence in ranked_contours:
            is_mold, cnn_confidence = self.cnn_classifier.classify_region(image, contour)
            
            if is_mold:
                # Combine confidences
                combined_confidence = (trad_confidence + cnn_confidence * 100) / 2
                verified_contours.append((contour, combined_confidence, cnn_confidence))
        
        # Sort by combined confidence
        verified_contours.sort(key=lambda x: x[1], reverse=True)
        verified_contours = verified_contours[:top_n]
        
        print(f"  CNN verified {len(verified_contours)} mold regions")
        
        # Draw results
        result_image = image.copy()
        for contour, combined_conf, cnn_conf in verified_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)
            
            # Draw circle
            cv2.circle(result_image, (x, y), radius, (0, 0, 255), 2)
            # Draw combined confidence
            cv2.putText(result_image, f"{combined_conf:.0f}%", 
                       (x - 20, y - radius - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Draw CNN confidence
            cv2.putText(result_image, f"CNN:{cnn_conf:.2f}", 
                       (x - 30, y + radius + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add header
        mold_found = len(verified_contours) > 0
        if mold_found:
            cv2.putText(result_image, f"MOLD DETECTED ({len(verified_contours)} regions)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(result_image, "No Mold Detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        output_path = f"{self.output_dir}/cnn_verified/{Path(image_path).name}"
        cv2.imwrite(output_path, result_image)
        
        return {
            'image_path': image_path,
            'mold_found': mold_found,
            'num_regions': len(verified_contours),
            'regions': verified_contours
        }
    
    def process_all(self, top_n=5):
        """
        Process all images in input directory.
        
        Args:
            top_n: Number of top detections to keep per image
            
        Returns:
            summary: Dictionary with overall results
        """
        image_files = list(Path(self.input_dir).glob('*.jpg')) + \
                     list(Path(self.input_dir).glob('*.png'))
        
        results = []
        
        print(f"\nProcessing {len(image_files)} images...")
        print("=" * 60)
        
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing {img_path.name}...")
            
            if self.use_cnn and self.cnn_classifier:
                # Process with CNN verification
                result = self.process_with_cnn(img_path, top_n=top_n)
            else:
                # Process with traditional detector only
                result = self.detector.process_image(img_path)
            
            results.append(result)
        
        # Generate summary
        mold_detected_count = sum(1 for r in results if r['mold_found'])
        total_regions = sum(r['num_regions'] for r in results)
        
        summary = {
            'total_images': len(image_files),
            'mold_detected': mold_detected_count,
            'clean_images': len(image_files) - mold_detected_count,
            'total_regions': total_regions,
            'results': results
        }
        
        # Save summary report
        self.save_report(summary)
        
        return summary
    
    def save_report(self, summary):
        """Save detection report to file."""
        report_path = f"{self.output_dir}/detection_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MOLD DETECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total Images Processed: {summary['total_images']}\n")
            f.write(f"Images with Mold: {summary['mold_detected']}\n")
            f.write(f"Clean Images: {summary['clean_images']}\n")
            f.write(f"Total Mold Regions: {summary['total_regions']}\n")
            f.write(f"Detection Method: {'CNN + Traditional' if self.use_cnn else 'Traditional Only'}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("INDIVIDUAL RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            for result in summary['results']:
                f.write(f"{Path(result['image_path']).name}:\n")
                f.write(f"  Mold Detected: {'YES' if result['mold_found'] else 'NO'}\n")
                f.write(f"  Regions: {result['num_regions']}\n\n")
        
        print(f"\nReport saved to: {report_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Mold Detection Pipeline')
    parser.add_argument('--input', default='resources/Photos-Mold', 
                       help='Input directory')
    parser.add_argument('--output', default='results/mold_detection', 
                       help='Output directory')
    parser.add_argument('--use-cnn', action='store_true', 
                       help='Use CNN for verification')
    parser.add_argument('--cnn-model', default='mold_cnn_model.pth', 
                       help='Path to CNN model')
    parser.add_argument('--top-n', type=int, default=5, 
                       help='Number of top detections per image')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = MoldDetectionPipeline(
        input_dir=args.input,
        output_dir=args.output,
        use_cnn=args.use_cnn,
        cnn_model_path=args.cnn_model if args.use_cnn else None
    )
    
    # Process all images
    summary = pipeline.process_all(top_n=args.top_n)
    
    # Print summary
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE!")
    print("=" * 60)
    print(f"Total Images: {summary['total_images']}")
    print(f"Mold Detected: {summary['mold_detected']}")
    print(f"Total Regions: {summary['total_regions']}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()