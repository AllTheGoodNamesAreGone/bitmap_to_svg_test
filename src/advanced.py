#!/usr/bin/env python3
"""
Advanced LayoutParser implementation specifically for question papers
Includes table structure analysis, text extraction, and SVG preparation
"""

import layoutparser as lp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path

class AdvancedQuestionPaperAnalyzer:
    def __init__(self):
        """Initialize with multiple specialized models"""
        # Layout detection model
        self.layout_model = lp.Detectron2LayoutModel(
            config_path='models\publaynet_mask_rcnn_R_50_FPN_3x/config.yml',
            model_path='models\publaynet_mask_rcnn_R_50_FPN_3x/model_final.pth',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        
        # Table detection model for better table handling
        self.table_model = lp.Detectron2LayoutModel(
            config_path='models/tableBank_faster_rcnn_R_50_FPN_3x/config.yaml',
            model_path='models/tableBank_faster_rcnn_R_50_FPN_3x/model_final.pth',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2],
            label_map={0: "Table"}
        )
        
        # OCR model for text extraction
        try:
            self.ocr_agent = lp.TesseractAgent(languages='eng')
        except:
            print("Warning: Tesseract not available. Text extraction will be skipped.")
            self.ocr_agent = None
    
    def analyze_question_paper(self, image_path, extract_text=True, analyze_tables=True):
        """
        Comprehensive analysis of question paper
        
        Args:
            image_path: Path to question paper image
            extract_text: Whether to extract text from regions
            analyze_tables: Whether to perform detailed table analysis
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {
            'image_info': {
                'path': image_path,
                'dimensions': image_rgb.shape,
                'filename': Path(image_path).name
            },
            'layout_blocks': [],
            'tables': [],
            'text_content': {},
            'structure_analysis': {}
        }
        
        # Step 1: General layout detection
        print("Detecting general layout...")
        layout_blocks = self.layout_model.detect(image_rgb)
        
        # Step 2: Enhanced table detection
        if analyze_tables:
            print("Performing enhanced table detection...")
            table_blocks = self.table_model.detect(image_rgb)
            results['tables'] = self.analyze_table_structures(image_rgb, table_blocks)
        
        # Step 3: Process layout blocks
        results['layout_blocks'] = self.process_layout_blocks(layout_blocks, image_rgb)
        
        # Step 4: Extract text content
        if extract_text and self.ocr_agent:
            print("Extracting text content...")
            results['text_content'] = self.extract_text_from_blocks(
                image_rgb, layout_blocks
            )
        
        # Step 5: Structural analysis for question paper format
        results['structure_analysis'] = self.analyze_question_paper_structure(
            layout_blocks, results['text_content']
        )
        
        return results
    
    def process_layout_blocks(self, layout_blocks, image):
        """Enhanced processing of layout blocks with additional features"""
        processed_blocks = []
        
        for i, block in enumerate(layout_blocks):
            # Basic block information
            block_info = {
                'id': i,
                'type': block.type,
                'confidence': float(block.score),
                'coordinates': {
                    'x1': int(block.coordinates[0]),
                    'y1': int(block.coordinates[1]),
                    'x2': int(block.coordinates[2]),
                    'y2': int(block.coordinates[3])
                }
            }
            
            # Calculate additional properties
            width = block_info['coordinates']['x2'] - block_info['coordinates']['x1']
            height = block_info['coordinates']['y2'] - block_info['coordinates']['y1']
            
            block_info.update({
                'dimensions': {'width': width, 'height': height},
                'area': width * height,
                'aspect_ratio': width / height if height > 0 else 0,
                'center': {
                    'x': (block_info['coordinates']['x1'] + block_info['coordinates']['x2']) // 2,
                    'y': (block_info['coordinates']['y1'] + block_info['coordinates']['y2']) // 2
                }
            })
            
            # Extract visual features (for SVG styling)
            block_image = self.extract_block_image(image, block.coordinates)
            block_info['visual_features'] = self.analyze_visual_features(block_image)
            
            processed_blocks.append(block_info)
        
        # Sort blocks by reading order (top to bottom, left to right)
        processed_blocks.sort(key=lambda x: (x['coordinates']['y1'], x['coordinates']['x1']))
        
        return processed_blocks
    
    def analyze_table_structures(self, image, table_blocks):
        """Detailed analysis of table structures"""
        table_analyses = []
        
        for i, table_block in enumerate(table_blocks):
            # Extract table region
            x1, y1, x2, y2 = map(int, table_block.coordinates)
            table_image = image[y1:y2, x1:x2]
            
            # Analyze table structure
            table_info = {
                'table_id': i,
                'confidence': float(table_block.score),
                'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'dimensions': {'width': x2-x1, 'height': y2-y1},
                'structure': self.detect_table_structure(table_image),
                'border_analysis': self.analyze_table_borders(table_image)
            }
            
            table_analyses.append(table_info)
        
        return table_analyses
    
    def detect_table_structure(self, table_image):
        """Detect rows and columns in table"""
        # Convert to grayscale
        gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Count approximate rows and columns
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50, 
                                  minLineLength=30, maxLineGap=10)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50, 
                                  minLineLength=30, maxLineGap=10)
        
        return {
            'estimated_rows': len(h_lines) - 1 if h_lines is not None else 'unknown',
            'estimated_cols': len(v_lines) - 1 if v_lines is not None else 'unknown',
            'has_borders': h_lines is not None or v_lines is not None
        }
    
    def analyze_table_borders(self, table_image):
        """Analyze table border styles for SVG recreation"""
        gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect border thickness and style
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        border_info = {
            'has_outer_border': len(contours) > 0,
            'border_complexity': len(contours),
            'estimated_thickness': 'medium'  # Could be enhanced with more analysis
        }
        
        return border_info
    
    def extract_text_from_blocks(self, image, layout_blocks):
        """Extract text content from each layout block"""
        text_content = {}
        
        for i, block in enumerate(layout_blocks):
            if block.type in ['Text', 'Title', 'List']:
                # Extract block image
                block_image = self.extract_block_image(image, block.coordinates)
                
                # Perform OCR
                try:
                    text = self.ocr_agent.detect(block_image)
                    text_content[f'block_{i}'] = {
                        'type': block.type,
                        'text': text,
                        'coordinates': block.coordinates
                    }
                except Exception as e:
                    print(f"OCR failed for block {i}: {e}")
                    text_content[f'block_{i}'] = {
                        'type': block.type,
                        'text': '',
                        'error': str(e)
                    }
        
        return text_content
    
    def analyze_question_paper_structure(self, layout_blocks, text_content):
        """Analyze the specific structure of question papers"""
        structure = {
            'header_detected': False,
            'question_blocks': [],
            'table_questions': [],
            'footer_detected': False,
            'total_questions': 0,
            'layout_type': 'unknown'
        }
        
        # Detect header (usually at top with institution info)
        if layout_blocks:
            top_blocks = [b for b in layout_blocks if b.coordinates[1] < 100]  # Top 100 pixels
            if top_blocks:
                structure['header_detected'] = True
        
        # Identify question patterns in text
        question_indicators = ['Q.No', 'Question', 'Q:', 'Marks:', 'CO', 'Blooms']
        
        for block_id, content in text_content.items():
            text = content.get('text', '').lower()
            if any(indicator.lower() in text for indicator in question_indicators):
                structure['question_blocks'].append({
                    'block_id': block_id,
                    'type': content['type'],
                    'contains_questions': True
                })
        
        # Analyze layout pattern
        structure['layout_type'] = self.determine_layout_pattern(layout_blocks)
        
        return structure
    
    def determine_layout_pattern(self, layout_blocks):
        """Determine the overall layout pattern of the question paper"""
        if not layout_blocks:
            return 'empty'
        
        # Count different types
        type_counts = {}
        for block in layout_blocks:
            type_counts[block.type] = type_counts.get(block.type, 0) + 1
        
        # Determine pattern based on composition
        if type_counts.get('Table', 0) > 2:
            return 'table_heavy'
        elif type_counts.get('Text', 0) > type_counts.get('Title', 0) * 3:
            return 'text_heavy'
        else:
            return 'mixed_content'
    
    def extract_block_image(self, image, coordinates):
        """Extract image region for a specific block"""
        x1, y1, x2, y2 = map(int, coordinates)
        return image[y1:y2, x1:x2]
    
    def analyze_visual_features(self, block_image):
        """Analyze visual features for SVG styling"""
        if block_image.size == 0:
            return {}
        
        # Basic color analysis
        mean_color = np.mean(block_image, axis=(0,1))
        
        # Detect if it's mostly text or graphics
        gray = cv2.cvtColor(block_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        return {
            'dominant_color': mean_color.tolist(),
            'edge_density': float(edge_density),
            'content_type': 'text' if edge_density < 0.1 else 'mixed'
        }
    
    def save_comprehensive_results(self, results, output_dir):
        """Save all analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = Path(results['image_info']['filename']).stem
        
        # Save JSON results
        with open(output_path / f"{filename}_analysis.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self.create_summary_report(results, output_path / f"{filename}_summary.txt")
        
        print(f"Comprehensive analysis saved to: {output_path}")
    
    def create_summary_report(self, results, output_file):
        """Create a human-readable summary report"""
        with open(output_file, 'w') as f:
            f.write(f"QUESTION PAPER ANALYSIS SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            
            f.write(f"Image: {results['image_info']['filename']}\n")
            f.write(f"Dimensions: {results['image_info']['dimensions']}\n\n")
            
            f.write(f"LAYOUT ANALYSIS:\n")
            f.write(f"Total blocks detected: {len(results['layout_blocks'])}\n")
            
            # Block type summary
            type_counts = {}
            for block in results['layout_blocks']:
                block_type = block['type']
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
            
            for block_type, count in type_counts.items():
                f.write(f"  {block_type}: {count}\n")
            
            f.write(f"\nTABLE ANALYSIS:\n")
            f.write(f"Tables detected: {len(results['tables'])}\n")
            
            f.write(f"\nSTRUCTURE ANALYSIS:\n")
            structure = results['structure_analysis']
            f.write(f"Header detected: {structure['header_detected']}\n")
            f.write(f"Layout type: {structure['layout_type']}\n")
            f.write(f"Question blocks found: {len(structure['question_blocks'])}\n")

# Example usage
if __name__ == "__main__":
    # Initialize advanced analyzer
    analyzer = AdvancedQuestionPaperAnalyzer()
    
    # Analyze a question paper
    image_path = "images/sample1.jpg"  # Replace with actual path
    
    try:
        print("Starting comprehensive analysis...")
        results = analyzer.analyze_question_paper(
            image_path, 
            extract_text=True, 
            analyze_tables=True
        )
        
        # Save comprehensive results
        analyzer.save_comprehensive_results(results, "comprehensive_output")
        
        # Print key findings
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        print(f"\nImage: {results['image_info']['filename']}")
        print(f"Layout blocks detected: {len(results['layout_blocks'])}")
        print(f"Tables detected: {len(results['tables'])}")
        print(f"Layout type: {results['structure_analysis']['layout_type']}")
        
        # Show block breakdown
        type_counts = {}
        for block in results['layout_blocks']:
            block_type = block['type']
            type_counts[block_type] = type_counts.get(block_type, 0) + 1
        
        print(f"\nBlock breakdown:")
        for block_type, count in type_counts.items():
            print(f"  {block_type}: {count}")
        
        # Show question structure insights
        structure = results['structure_analysis']
        if structure['question_blocks']:
            print(f"\nQuestion paper structure:")
            print(f"  Header detected: {structure['header_detected']}")
            print(f"  Question blocks found: {len(structure['question_blocks'])}")
            print(f"  Table-based questions: {len(structure['table_questions'])}")
        
        print(f"\nDetailed results saved to: comprehensive_output/")
        
    except FileNotFoundError:
        print("Please replace 'your_question_paper.jpg' with the actual path to your image")
        print("\nFor batch processing multiple images:")
        print("image_paths = ['paper1.jpg', 'paper2.jpg', 'paper3.jpg']")
        print("for path in image_paths:")
        print("    results = analyzer.analyze_question_paper(path)")
        print("    analyzer.save_comprehensive_results(results, f'output_{Path(path).stem}')")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you have installed all dependencies:")
        print("  pip install layoutparser torchvision")
        print("  pip install 'layoutparser[paddledetection]'")
        print("  pip install detectron2")
        print("  pip install tesseract (for OCR)")

# Additional utility functions for SVG preparation
class SVGPreparationHelper:
    """Helper class to prepare layout analysis results for SVG conversion"""
    
    @staticmethod
    def prepare_svg_structure(analysis_results):
        """Convert layout analysis to SVG-ready structure"""
        svg_elements = []
        
        for block in analysis_results['layout_blocks']:
            element = {
                'type': 'rect' if block['type'] in ['Table', 'Figure'] else 'text',
                'id': f"block_{block['id']}",
                'class': block['type'].lower(),
                'coordinates': block['coordinates'],
                'dimensions': block['dimensions'],
                'style': SVGPreparationHelper.determine_svg_style(block),
                'content': block.get('text_content', ''),
                'z_index': SVGPreparationHelper.calculate_z_index(block['type'])
            }
            svg_elements.append(element)
        
        return {
            'viewport': {
                'width': analysis_results['image_info']['dimensions'][1],
                'height': analysis_results['image_info']['dimensions'][0]
            },
            'elements': svg_elements,
            'tables': SVGPreparationHelper.prepare_table_elements(analysis_results['tables'])
        }
    
    @staticmethod
    def determine_svg_style(block):
        """Determine SVG styling based on block properties"""
        styles = {
            'Title': {
                'font-family': 'serif',
                'font-weight': 'bold',
                'font-size': '16px',
                'fill': '#000000'
            },
            'Text': {
                'font-family': 'serif',
                'font-weight': 'normal',
                'font-size': '12px',
                'fill': '#000000'
            },
            'Table': {
                'stroke': '#000000',
                'stroke-width': '1',
                'fill': 'none'
            },
            'Figure': {
                'stroke': '#666666',
                'stroke-width': '1',
                'fill': '#f9f9f9'
            },
            'List': {
                'font-family': 'serif',
                'font-weight': 'normal',
                'font-size': '11px',
                'fill': '#000000'
            }
        }
        
        return styles.get(block['type'], styles['Text'])
    
    @staticmethod
    def calculate_z_index(block_type):
        """Calculate layering order for SVG elements"""
        z_indices = {
            'Figure': 1,
            'Table': 2,
            'Text': 3,
            'List': 3,
            'Title': 4
        }
        return z_indices.get(block_type, 3)
    
    @staticmethod
    def prepare_table_elements(tables):
        """Prepare table-specific SVG elements"""
        table_elements = []
        
        for table in tables:
            # Create table structure
            table_element = {
                'type': 'table',
                'id': f"table_{table['table_id']}",
                'coordinates': table['coordinates'],
                'dimensions': table['dimensions'],
                'structure': table['structure'],
                'border_style': table['border_analysis']
            }
            table_elements.append(table_element)
        
        return table_elements

# Batch processing utility
def batch_process_question_papers(image_directory, output_directory="batch_analysis"):
    """Process multiple question papers in a directory"""
    import os
    from pathlib import Path
    
    analyzer = AdvancedQuestionPaperAnalyzer()
    svg_helper = SVGPreparationHelper()
    
    image_dir = Path(image_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))
        image_files.extend(image_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    results_summary = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            # Analyze the image
            results = analyzer.analyze_question_paper(str(image_path))
            
            # Save individual results
            file_output_dir = output_dir / image_path.stem
            analyzer.save_comprehensive_results(results, file_output_dir)
            
            # Prepare SVG structure
            svg_structure = svg_helper.prepare_svg_structure(results)
            with open(file_output_dir / f"{image_path.stem}_svg_structure.json", 'w') as f:
                json.dump(svg_structure, f, indent=2)
            
            # Add to summary
            summary_entry = {
                'filename': image_path.name,
                'blocks_detected': len(results['layout_blocks']),
                'tables_detected': len(results['tables']),
                'layout_type': results['structure_analysis']['layout_type'],
                'header_detected': results['structure_analysis']['header_detected'],
                'status': 'success'
            }
            results_summary.append(summary_entry)
            
            print(f"  ✓ Completed - {len(results['layout_blocks'])} blocks, {len(results['tables'])} tables")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results_summary.append({
                'filename': image_path.name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save batch summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_dir / "batch_summary.csv", index=False)
    
    print(f"\n{'='*50}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Processed: {len(image_files)} images")
    print(f"Successful: {len([r for r in results_summary if r['status'] == 'success'])}")
    print(f"Failed: {len([r for r in results_summary if r['status'] == 'failed'])}")
    print(f"Results saved to: {output_dir}")
    
    return results_summary

# Example batch processing
# Uncomment and modify the path below to process multiple images
# batch_results = batch_process_question_papers("/path/to/your/question_papers", "batch_output")