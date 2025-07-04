#!/usr/bin/env python3
"""
PaddleOCR PP-Structure alternative for Windows
More reliable than LayoutParser on Windows systems
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

try:
    from paddleocr import PPStructure, draw_structure_result
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    print("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")

class PaddleOCRQuestionPaperAnalyzer:
    def __init__(self, use_gpu=False, lang='en'):
        """
        Initialize PaddleOCR PP-Structure analyzer
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('en' for English)
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
        
        # Initialize PP-Structure
        self.engine = PPStructure(
            use_gpu=use_gpu,
            show_log=True,
            lang=lang,
            layout=True,  # Enable layout analysis
            table=True,   # Enable table recognition
            ocr=True      # Enable text recognition
        )
        
        print("✓ PaddleOCR PP-Structure initialized successfully")
    
    def analyze_image(self, image_path, output_dir="paddle_output"):
        """
        Analyze question paper using PaddleOCR PP-Structure
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"Analyzing image: {image_path}")
        print("This may take a moment for the first run...")
        
        # Run PP-Structure analysis
        result = self.engine(img)
        
        # Process results
        processed_results = self.process_paddle_results(result, img.shape)
        
        # Save visualizations and data
        self.save_results(img, result, processed_results, image_path, output_dir)
        
        return processed_results, result
    
    def process_paddle_results(self, paddle_result, image_shape):
        """Convert PaddleOCR results to standardized format"""
        processed_blocks = []
        
        for i, item in enumerate(paddle_result):
            # Extract layout information
            layout_info = item.get('layout', {})
            
            if 'bbox' in layout_info:
                bbox = layout_info['bbox']
                
                block_info = {
                    'id': i,
                    'type': layout_info.get('label', 'unknown'),
                    'confidence': float(layout_info.get('score', 0.0)),
                    'coordinates': {
                        'x1': int(bbox[0]),
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]),
                        'y2': int(bbox[3])
                    },
                    'dimensions': {
                        'width': int(bbox[2] - bbox[0]),
                        'height': int(bbox[3] - bbox[1])
                    },
                    'area': int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                }
                
                # Add OCR text if available
                if 'res' in item:
                    text_results = []
                    for text_item in item['res']:
                        if isinstance(text_item, dict) and 'text' in text_item:
                            text_results.append(text_item['text'])
                        elif isinstance(text_item, list) and len(text_item) >= 2:
                            text_results.append(text_item[1][0] if text_item[1] else '')
                    
                    block_info['text_content'] = ' '.join(text_results)
                    block_info['has_text'] = len(text_results) > 0
                else:
                    block_info['text_content'] = ''
                    block_info['has_text'] = False
                
                # Special handling for tables
                if block_info['type'].lower() == 'table' and 'res' in item:
                    block_info['table_data'] = self.extract_table_data(item['res'])
                
                processed_blocks.append(block_info)
        
        # Sort by vertical position
        processed_blocks.sort(key=lambda x: x['coordinates']['y1'])
        
        return processed_blocks
    
    def extract_table_data(self, table_result):
        """Extract structured data from table results"""
        if not table_result:
            return {}
        
        # PaddleOCR table results are often in HTML format
        table_info = {
            'structure_detected': True,
            'content_type': 'table',
            'raw_result': table_result
        }
        
        # Try to extract table structure if available
        if isinstance(table_result, list) and table_result:
            first_result = table_result[0]
            if isinstance(first_result, dict):
                if 'html' in first_result:
                    table_info['html_structure'] = first_result['html']
                if 'bbox' in first_result:
                    table_info['cell_bboxes'] = first_result['bbox']
        
        return table_info
    
    def save_results(self, image, paddle_result, processed_results, image_path, output_dir):
        """Save analysis results and visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        
        # Save visualization
        try:
            # Draw structure results
            im_show = draw_structure_result(image, paddle_result, font_path=None)
            
            # Save annotated image
            plt.figure(figsize=(15, 20))
            plt.imshow(cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"PaddleOCR Structure Analysis: {base_name}")
            plt.tight_layout()
            
            viz_path = os.path.join(output_dir, f"{base_name}_paddle_analysis.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved visualization: {viz_path}")
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
        
        # Save structured data
        output_data = {
            'image_name': os.path.basename(image_path),
            'analysis_engine': 'PaddleOCR PP-Structure',
            'total_blocks': len(processed_results),
            'block_types': list(set([block['type'] for block in processed_results])),
            'blocks': processed_results,
            'raw_paddle_result': paddle_result  # Keep original for reference
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_paddle_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved JSON data: {json_path}")
        
        # Create summary
        self.print_analysis_summary(base_name, processed_results)
        
        # Save individual blocks
        self.save_individual_blocks(image, processed_results, output_dir, base_name)
    
    def save_individual_blocks(self, image, blocks, output_dir, base_name):
        """Save individual block images"""
        for block in blocks:
            try:
                coords = block['coordinates']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                
                if x2 > x1 and y2 > y1:  # Valid coordinates
                    block_image = image[y1:y2, x1:x2]
                    
                    if block_image.size > 0:
                        plt.figure(figsize=(8, 6))
                        plt.imshow(cv2.cvtColor(block_image, cv2.COLOR_BGR2RGB))
                        plt.axis('off')
                        
                        title = f"Block {block['id']}: {block['type']}"
                        if block.get('has_text', False):
                            preview_text = block['text_content'][:50] + "..." if len(block['text_content']) > 50 else block['text_content']
                            title += f"\nText: {preview_text}"
                        
                        plt.title(title, fontsize=10)
                        
                        block_path = os.path.join(output_dir, f"{base_name}_block_{block['id']}_{block['type']}.png")
                        plt.savefig(block_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
            except Exception as e:
                print(f"Warning: Could not save block {block['id']}: {e}")
                continue
    
    def print_analysis_summary(self, base_name, results):
        """Print analysis summary"""
        print(f"\n=== PaddleOCR Analysis Summary for {base_name} ===")
        print(f"Total blocks detected: {len(results)}")
        
        type_counts = {}
        text_blocks = 0
        
        for block in results:
            block_type = block['type']
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
            
            if block.get('has_text', False):
                text_blocks += 1
        
        print("Block types:")
        for block_type, count in type_counts.items():
            print(f"  {block_type}: {count}")
        
        print(f"Blocks with text content: {text_blocks}")
        
        # Show some extracted text
        text_blocks_sample = [b for b in results if b.get('has_text', False)][:3]
        if text_blocks_sample:
            print("\nSample extracted text:")
            for i, block in enumerate(text_blocks_sample, 1):
                preview = block['text_content'][:100] + "..." if len(block['text_content']) > 100 else block['text_content']
                print(f"  {i}. {preview}")
    
    def analyze_multiple_images(self, image_directory, output_directory="paddle_batch"):
        """Analyze multiple question papers"""
        image_dir = Path(image_directory)
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)
        
        # Find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images to process")
        
        results_summary = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
            
            try:
                results, raw_results = self.analyze_image(str(image_path), 
                                                        output_dir / image_path.stem)
                
                summary_entry = {
                    'filename': image_path.name,
                    'blocks_detected': len(results),
                    'text_blocks': len([r for r in results if r.get('has_text', False)]),
                    'table_blocks': len([r for r in results if r['type'].lower() == 'table']),
                    'status': 'success'
                }
                results_summary.append(summary_entry)
                
                print(f"  ✓ Completed - {len(results)} blocks detected")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results_summary.append({
                    'filename': image_path.name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Save batch summary
        import pandas as pd
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(output_dir / "batch_summary.csv", index=False)
        
        print(f"\nBatch processing complete. Results saved to: {output_dir}")
        return results_summary


# Example usage
if __name__ == "__main__":
    try:
        # Initialize PaddleOCR analyzer
        analyzer = PaddleOCRQuestionPaperAnalyzer(use_gpu=True)  # Set to True if you have GPU
        
        # Test with your image
        image_path = "images/sample1.jpg"  # Replace with actual path
        
        if os.path.exists(image_path):
            print(f"Analyzing: {image_path}")
            results, raw_results = analyzer.analyze_image(image_path)
            
            print(f"\n✓ Analysis completed!")
            print(f"Detected {len(results)} layout blocks")
            
            # Show detailed results
            print("\nDetailed block information:")
            for block in results:
                print(f"Block {block['id']}: {block['type']} (confidence: {block['confidence']:.3f})")
                if block.get('has_text', False):
                    preview = block['text_content'][:100] + "..." if len(block['text_content']) > 100 else block['text_content']
                    print(f"  Text: {preview}")
                print(f"  Size: {block['dimensions']['width']} x {block['dimensions']['height']}")
                print()
                
        else:
            print(f"Image not found: {image_path}")
            print("Please replace with the actual path to your question paper image")
            
    except ImportError:
        print("Please install PaddleOCR first:")
        print("pip install paddlepaddle paddleocr")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure PaddleOCR is properly installed and the image path is correct")