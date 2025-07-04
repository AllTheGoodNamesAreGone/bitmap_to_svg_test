#!/usr/bin/env python3
"""
LayoutParser Windows compatibility fix
This version handles Windows path issues and provides fallback options
"""

import layoutparser as lp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import tempfile
from pathlib import Path

class WindowsCompatibleQuestionPaperAnalyzer:
    def __init__(self, model_type="publaynet_faster_rcnn"):
        """
        Initialize with Windows-compatible paths and fallback options
        
        Args:
            model_type: Choose from 'publaynet_faster_rcnn', 'publaynet_mask_rcnn', 'primalayout'
        """
        # Set up Windows-compatible cache directory
        self.setup_cache_directory()
        
        self.model_configs = {
            'publaynet_faster_rcnn': {
                'config': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                'model': 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model',
                'labels': {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            },
            'publaynet_mask_rcnn': {
                'config': 'lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config',
                'model': 'lp://PubLayNet/mask_rcnn_R_50_FPN_3x/model',
                'labels': {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            },
            'primalayout': {
                'config': 'lp://PrimaLayout/faster_rcnn_R_50_FPN_3x/config',
                'model': 'lp://PrimaLayout/faster_rcnn_R_50_FPN_3x/model',
                'labels': {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            }
        }
        
        # Try to load model with multiple fallback strategies
        self.model = None
        self.load_model_with_fallbacks(model_type)
    
    def setup_cache_directory(self):
        """Set up a Windows-compatible cache directory"""
        # Create a simple cache directory in the project folder
        cache_dir = Path.cwd() / "layoutparser_cache"
        cache_dir.mkdir(exist_ok=True)
        
        # Set environment variables to use our cache
        os.environ['TORCH_HOME'] = str(cache_dir / "torch")
        os.environ['IOPATH_CACHE'] = str(cache_dir / "iopath")
        
        # Create subdirectories
        (cache_dir / "torch").mkdir(exist_ok=True)
        (cache_dir / "iopath").mkdir(exist_ok=True)
        
        print(f"Using cache directory: {cache_dir}")
    
    def load_model_with_fallbacks(self, model_type):
        """Try multiple strategies to load the model"""
        if model_type not in self.model_configs:
            raise ValueError(f"Model type {model_type} not supported")
        
        config = self.model_configs[model_type]
        
        # Strategy 1: Try with lower confidence threshold and simpler config
        strategies = [
            {"extra_config": ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]},
            {"extra_config": ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7]},
            {"extra_config": []},  # No extra config
        ]
        
        for i, strategy in enumerate(strategies, 1):
            print(f"Trying loading strategy {i}...")
            try:
                self.model = lp.Detectron2LayoutModel(
                    config_path='models\publaynet_mask_rcnn_R_50_FPN_3x/config.yml',
                    model_path='models\publaynet_mask_rcnn_R_50_FPN_3x/model_final.pth',
                    extra_config=strategy.get("extra_config", []),
                    label_map=config['labels']
                )
                print(f"✓ Successfully loaded {model_type} model with strategy {i}")
                return
                
            except Exception as e:
                print(f"✗ Strategy {i} failed: {str(e)[:100]}...")
                continue
        
        # If all strategies fail, try alternative approach
        print("All standard strategies failed. Trying alternative models...")
        self.try_alternative_models()
    
    def try_alternative_models(self):
        """Try simpler, more reliable models as fallbacks"""
        alternative_configs = [
            {
                'name': 'PrimaLayout Faster R-CNN',
                'config': 'lp://PrimaLayout/faster_rcnn_R_50_FPN_3x/config',
                'model': 'lp://PrimaLayout/faster_rcnn_R_50_FPN_3x/model',
                'labels': {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
            }
        ]
        
        for alt_config in alternative_configs:
            try:
                print(f"Trying {alt_config['name']}...")
                self.model = lp.Detectron2LayoutModel(
                    config_path=alt_config['config'],
                    model_path=alt_config['model'],
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.6],
                    label_map=alt_config['labels']
                )
                print(f"✓ Successfully loaded {alt_config['name']}")
                return
            except Exception as e:
                print(f"✗ {alt_config['name']} failed: {str(e)[:100]}...")
                continue
        
        # If everything fails, provide manual download instructions
        raise RuntimeError(
            "Could not load any LayoutParser models automatically. "
            "This is likely due to Windows path issues. "
            "Please try the manual download method below."
        )
    
    def analyze_image(self, image_path, output_dir="output"):
        """
        Analyze a question paper image (same as before, but with error handling)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Please check the initialization errors above.")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect layout with error handling
        print("Detecting layout...")
        try:
            layout_blocks = self.model.detect(image_rgb)
            print(f"✓ Detected {len(layout_blocks)} layout blocks")
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], []
        
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
        os.makedirs(output_dir, exist_ok=True)
        
        # Create annotated image
        try:
            annotated_image = lp.draw_box(
                image.copy(), 
                layout_blocks, 
                box_width=3, 
                show_element_type=True,
                show_element_id=True
            )
        except Exception as e:
            print(f"Warning: Could not create annotated image: {e}")
            annotated_image = image.copy()
        
        # Save full annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        plt.figure(figsize=(15, 20))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.title(f"Layout Analysis: {base_name}")
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"{base_name}_layout_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved annotated image: {output_path}")
        
        # Save individual blocks
        for i, block in enumerate(layout_blocks):
            try:
                x1, y1, x2, y2 = map(int, block.coordinates)
                if x2 > x1 and y2 > y1:  # Valid coordinates
                    block_image = image[y1:y2, x1:x2]
                    
                    if block_image.size > 0:
                        plt.figure(figsize=(8, 6))
                        plt.imshow(block_image)
                        plt.axis('off')
                        plt.title(f"Block {i}: {block.type} (confidence: {block.score:.2f})")
                        
                        block_path = os.path.join(output_dir, f"{base_name}_block_{i}_{block.type}.png")
                        plt.savefig(block_path, dpi=150, bbox_inches='tight')
                        plt.close()
            except Exception as e:
                print(f"Warning: Could not save block {i}: {e}")
                continue
    
    def save_structured_data(self, results, output_dir, image_path):
        """Save structured data as JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        output_data = {
            'image_name': os.path.basename(image_path),
            'total_blocks': len(results),
            'block_types': list(set([block['type'] for block in results])),
            'blocks': results
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_layout_data.json")
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved JSON data: {json_path}")
        
        # Create summary
        print(f"\n=== Layout Analysis Summary for {base_name} ===")
        print(f"Total blocks detected: {len(results)}")
        
        type_counts = {}
        for block in results:
            block_type = block['type']
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        for block_type, count in type_counts.items():
            print(f"{block_type}: {count} blocks")


# Manual download fallback method
def manual_model_setup():
    """
    Instructions for manual model setup if automatic download fails
    """
    print("""
    MANUAL MODEL SETUP INSTRUCTIONS
    ===============================
    
    If automatic model loading fails, you can download models manually:
    
    1. Create a models directory in your project:
       mkdir models
    
    2. Download model files manually:
       - Go to: https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/models/detectron2/catalog.py
       - Find the direct download URLs for the models
       - Download config.yaml and model.pth files
    
    3. Use local paths instead of 'lp://' URLs:
       model = lp.Detectron2LayoutModel(
           config_path='./models/config.yaml',
           model_path='./models/model.pth',
           label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
       )
    
    4. Alternative: Use PaddleOCR PP-Structure (often more reliable on Windows):
       pip install paddlepaddle paddleocr
       
       from paddleocr import PPStructure
       engine = PPStructure(show_log=True)
       result = engine(img_path)
    """)


# Example usage with better error handling
if __name__ == "__main__":
    try:
        print("Initializing Windows-compatible analyzer...")
        analyzer = WindowsCompatibleQuestionPaperAnalyzer(model_type="publaynet_faster_rcnn")
        
        # Test with your image
        image_path = "images/sample1.jpg"  # Replace with actual path
        
        if os.path.exists(image_path):
            print(f"Analyzing: {image_path}")
            results, layout_blocks = analyzer.analyze_image(image_path)
            
            if results:
                print(f"\n✓ Analysis completed successfully!")
                print(f"Found {len(results)} layout blocks")
                
                # Show results
                for block in results[:5]:  # Show first 5 blocks
                    print(f"  {block['type']}: confidence {block['confidence']:.2f}")
            else:
                print("No layout blocks detected or analysis failed")
        else:
            print(f"Image not found: {image_path}")
            print("Please replace with the actual path to your question paper image")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative approach...")
        manual_model_setup()
        
        # Fallback to PaddleOCR approach
        print("\nFALLBACK: Try PaddleOCR PP-Structure instead:")
        print("pip install paddlepaddle paddleocr")
        print("Then use the PaddleOCR approach shown in the manual setup instructions")