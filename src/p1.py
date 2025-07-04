#!/usr/bin/env python3
"""
Basic LayoutParser implementation for question paper analysis
This script demonstrates the core functionality you'll need
"""

import layoutparser as lp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

class QuestionPaperAnalyzer:
    def __init__(self, model_type="publaynet_mask_rcnn"):
        """
        Initialize the layout analyzer
        
        Args:
            model_type: Choose from 'publaynet_mask_rcnn', 'publaynet_faster_rcnn', 'primalayout'
        """
        self.model_configs = {
            'publaynet_mask_rcnn': {
                'config': 'models/publaynet_mask_rcnn_X_101_32x8d_FPN_3x/config.yaml',
                'model': 'models/publaynet_mask_rcnn_X_101_32x8d_FPN_3x/model_final.pth',
                'labels': {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            },
            'publaynet_faster_rcnn': {
                'config': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                'model': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model',
                'labels': {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            },
            'tablebank': {
                'config': 'lp://TableBank/faster_rcnn_R_50_FPN_3x/config',
                'model': 'lp://TableBank/faster_rcnn_R_50_FPN_3x/model',
                'labels': {0: "Table"}
            }
        }
        
        self.load_model(model_type)
    
    def load_model(self, model_type):
        """Load the specified model"""
        if model_type not in self.model_configs:
            raise ValueError(f"Model type {model_type} not supported")
        
        config = self.model_configs[model_type]
        
        self.model = lp.Detectron2LayoutModel(
            config_path=config['config'],
            model_path=config['model'],
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],  # Confidence threshold
            label_map=config['labels']
        )
        
        print(f"Loaded {model_type} model successfully")
    
    def analyze_image(self, image_path, output_dir="output"):
        """
        Analyze a question paper image
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect layout
        print("Detecting layout...")
        layout_blocks = self.model.detect(image_rgb)
        
        # Process results
        results = self.process_layout_blocks(layout_blocks, image_rgb.shape)
        
        # Save visualizations
        self.save_visualizations(image_rgb, layout_blocks, output_dir, image_path)
        
        # Save structured data
        self.save_structured_data(results, output_dir, image_path)
        
        return results, layout_blocks
    
    def process_layout_blocks(self, layout_blocks, image_shape):
        """Process detected layout blocks into structured format"""
        results = []
        
        for i, block in enumerate(layout_blocks):
            block_info = {
                'id': i,
                'type': block.type,
                'confidence': float(block.score),
                'coordinates': {
                    'x1': int(block.coordinates[0]),
                    'y1': int(block.coordinates[1]),
                    'x2': int(block.coordinates[2]),
                    'y2': int(block.coordinates[3])
                },
                'dimensions': {
                    'width': int(block.coordinates[2] - block.coordinates[0]),
                    'height': int(block.coordinates[3] - block.coordinates[1])
                },
                'area': int((block.coordinates[2] - block.coordinates[0]) * 
                           (block.coordinates[3] - block.coordinates[1]))
            }
            results.append(block_info)
        
        # Sort by vertical position (top to bottom)
        results.sort(key=lambda x: x['coordinates']['y1'])
        
        return results
    
    def save_visualizations(self, image, layout_blocks, output_dir, image_path):
        """Save visualization images"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create annotated image
        annotated_image = lp.draw_box(
            image.copy(), 
            layout_blocks, 
            box_width=3, 
            show_element_type=True,
            show_element_id=True
        )
        
        # Save full annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        plt.figure(figsize=(15, 20))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title(f"Layout Analysis: {base_name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_layout_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save individual blocks
        for i, block in enumerate(layout_blocks):
            x1, y1, x2, y2 = map(int, block.coordinates)
            block_image = image[y1:y2, x1:x2]
            
            if block_image.size > 0:
                plt.figure(figsize=(8, 6))
                plt.imshow(block_image)
                plt.axis('off')
                plt.title(f"Block {i}: {block.type} (confidence: {block.score:.2f})")
                plt.savefig(f"{output_dir}/{base_name}_block_{i}_{block.type}.png", 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    def save_structured_data(self, results, output_dir, image_path):
        """Save structured data as JSON"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        output_data = {
            'image_name': os.path.basename(image_path),
            'total_blocks': len(results),
            'block_types': list(set([block['type'] for block in results])),
            'blocks': results
        }
        
        with open(f"{output_dir}/{base_name}_layout_data.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Create summary
        print(f"\n=== Layout Analysis Summary for {base_name} ===")
        print(f"Total blocks detected: {len(results)}")
        
        type_counts = {}
        for block in results:
            block_type = block['type']
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        for block_type, count in type_counts.items():
            print(f"{block_type}: {count} blocks")
        
        print(f"Results saved to: {output_dir}/")
    
    def analyze_multiple_images(self, image_paths, output_dir="batch_output"):
        """Analyze multiple question papers"""
        all_results = {}
        
        for image_path in image_paths:
            print(f"\nAnalyzing: {image_path}")
            try:
                results, blocks = self.analyze_image(image_path, output_dir)
                all_results[image_path] = results
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
        
        return all_results

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = QuestionPaperAnalyzer(model_type="publaynet_mask_rcnn")
    
    # Test with a single image
    # Replace with your question paper image path
    image_path = "images/sample1.jpg"
    
    try:
        results, layout_blocks = analyzer.analyze_image(image_path)
        
        # Print detailed results
        print("\n=== Detailed Block Information ===")
        for block in results:
            print(f"Block {block['id']}: {block['type']}")
            print(f"  Confidence: {block['confidence']:.3f}")
            print(f"  Position: ({block['coordinates']['x1']}, {block['coordinates']['y1']}) to "
                  f"({block['coordinates']['x2']}, {block['coordinates']['y2']})")
            print(f"  Size: {block['dimensions']['width']} x {block['dimensions']['height']}")
            print(f"  Area: {block['area']} pixels")
            print()
            
    except FileNotFoundError:
        print("Please replace 'your_question_paper.jpg' with the actual path to your image")
        print("Example usage:")
        print("  analyzer.analyze_image('/path/to/your/question_paper.jpg')")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all dependencies and the image path is correct")