#!/usr/bin/env python3
"""
Custom Table Parser specifically for question paper formats
Focuses on extracting table structure at cell level
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import pytesseract
from pathlib import Path
import os 

class QuestionPaperTableParser:
    def __init__(self):
        """Initialize custom table parser"""
        self.cell_data = []
        self.horizontal_lines = []
        self.vertical_lines = []
        
    def parse_question_paper_table(self, image_path):
        """Parse question paper table structure at cell level"""
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"🔍 Parsing table structure in {image_path}")
        
        # Step 1: Detect table lines with multiple methods
        h_lines, v_lines = self.detect_table_lines_advanced(gray)
        
        # Step 2: Create grid from line intersections
        grid_cells = self.create_cell_grid(h_lines, v_lines, image_rgb.shape)
        
        # Step 3: Extract content from each cell
        cell_contents = self.extract_cell_contents(image_rgb, grid_cells)
        
        # Step 4: Classify cells by content type
        classified_cells = self.classify_cells(cell_contents)
        
        # Step 5: Create SVG structure
        svg_data = self.create_table_svg_structure(classified_cells, image_rgb.shape)
        
        # Step 6: Visualize results
        self.visualize_table_parsing(image_rgb, classified_cells, h_lines, v_lines, image_path)
        
        # Step 7: Save detailed results
        self.save_table_parsing_results(svg_data, classified_cells, image_path)
        
        return classified_cells, svg_data
    
    def detect_table_lines_advanced(self, gray_image):
        """Advanced table line detection using multiple methods"""
        
        # Method 1: Morphological operations
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))  # Longer for better detection
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
        
        horizontal_lines_morph = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, h_kernel)
        vertical_lines_morph = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, v_kernel)
        
        # Method 2: Edge detection + Hough transform
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=100, maxLineGap=10)
        
        # Extract horizontal and vertical lines from Hough
        h_lines_hough = []
        v_lines_hough = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Classify as horizontal or vertical
                if abs(y2 - y1) < 10:  # Horizontal (small y difference)
                    h_lines_hough.append((min(y1, y2), max(x1, x2) - min(x1, x2)))  # (y_pos, length)
                elif abs(x2 - x1) < 10:  # Vertical (small x difference)
                    v_lines_hough.append((min(x1, x2), max(y1, y2) - min(y1, y2)))  # (x_pos, length)
        
        # Method 3: Extract line positions from morphological results
        h_lines_morph_pos = self.extract_line_positions(horizontal_lines_morph, axis='horizontal')
        v_lines_morph_pos = self.extract_line_positions(vertical_lines_morph, axis='vertical')
        
        # Combine and deduplicate
        h_lines_combined = self.combine_line_detections(h_lines_morph_pos, h_lines_hough, axis='horizontal')
        v_lines_combined = self.combine_line_detections(v_lines_morph_pos, v_lines_hough, axis='vertical')
        
        print(f"   Detected {len(h_lines_combined)} horizontal lines, {len(v_lines_combined)} vertical lines")
        
        return h_lines_combined, v_lines_combined
    
    def extract_line_positions(self, line_image, axis):
        """Extract line positions from morphological operation results"""
        if axis == 'horizontal':
            line_sums = np.sum(line_image, axis=1)  # Sum across width
        else:
            line_sums = np.sum(line_image, axis=0)  # Sum across height
        
        # Find peaks
        threshold = np.max(line_sums) * 0.3
        positions = []
        
        in_line = False
        line_start = 0
        
        for i, val in enumerate(line_sums):
            if val > threshold and not in_line:
                line_start = i
                in_line = True
            elif val <= threshold and in_line:
                # End of line, record middle position
                positions.append((line_start + i) // 2)
                in_line = False
        
        return positions
    
    def combine_line_detections(self, morph_lines, hough_lines, axis):
        """Combine morphological and Hough line detections"""
        all_positions = list(morph_lines)
        
        # Add Hough lines that are significant
        for pos, length in hough_lines:
            if length > 50:  # Only significant lines
                all_positions.append(pos)
        
        # Remove duplicates (merge nearby lines)
        if not all_positions:
            return []
        
        all_positions.sort()
        merged = [all_positions[0]]
        
        for pos in all_positions[1:]:
            if pos - merged[-1] > 15:  # Minimum distance between lines
                merged.append(pos)
            else:
                # Merge with previous (take average)
                merged[-1] = (merged[-1] + pos) // 2
        
        return merged
    
    def create_cell_grid(self, h_lines, v_lines, image_shape):
        """Create grid of cells from detected lines"""
        cells = []
        
        # Add image boundaries to lines
        h_lines_with_bounds = [0] + h_lines + [image_shape[0]]
        v_lines_with_bounds = [0] + v_lines + [image_shape[1]]
        
        h_lines_with_bounds.sort()
        v_lines_with_bounds.sort()
        
        # Create cells from grid intersections
        for i in range(len(h_lines_with_bounds) - 1):
            for j in range(len(v_lines_with_bounds) - 1):
                y1 = h_lines_with_bounds[i]
                y2 = h_lines_with_bounds[i + 1]
                x1 = v_lines_with_bounds[j]
                x2 = v_lines_with_bounds[j + 1]
                
                # Filter out very small cells
                if (x2 - x1) > 20 and (y2 - y1) > 10:
                    cells.append({
                        'row': i,
                        'col': j,
                        'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        print(f"   Created grid with {len(cells)} cells")
        return cells
    
    def extract_cell_contents(self, image, cells):
        """Extract text content from each cell using OCR"""
        
        for i, cell in enumerate(cells):
            coords = cell['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Add padding to avoid cutting text
            padding = 2
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Extract cell image
            cell_image = image[y1:y2, x1:x2]
            
            # OCR extraction
            try:
                text = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
                
                # Get detailed OCR data for confidence
                ocr_data = pytesseract.image_to_data(cell_image, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                cell['text_content'] = text
                cell['ocr_confidence'] = avg_confidence
                cell['has_text'] = len(text.strip()) > 0
                
            except Exception as e:
                cell['text_content'] = ''
                cell['ocr_confidence'] = 0
                cell['has_text'] = False
                print(f"   ⚠️  OCR failed for cell {i}: {e}")
        
        return cells
    
    def classify_cells(self, cells):
        """Classify cells by content type and purpose"""
        
        for cell in cells:
            text = cell['text_content'].lower().strip()
            
            # Classification logic
            if not cell['has_text']:
                cell['cell_type'] = 'empty'
            elif any(keyword in text for keyword in ['q.no', 'questions', 'blooms', 'levels', 'co', 'marks']):
                cell['cell_type'] = 'header'
            elif text.startswith(('1.', '2.', '3.', '1a', '1b', '2a', '2b')):
                cell['cell_type'] = 'question_number'
            elif any(keyword in text for keyword in ['discuss', 'explain', 'describe', 'differentiate', 'derive']):
                cell['cell_type'] = 'question_text'
            elif text.isdigit() or (len(text) <= 5 and any(c.isdigit() for c in text)):
                cell['cell_type'] = 'score_value'
            elif text in ['l1', 'l2', 'l3', 'co1', 'co2', 'co3', 'co4', 'co5']:
                cell['cell_type'] = 'classification_code'
            elif len(text) > 50:
                cell['cell_type'] = 'long_text'
            else:
                cell['cell_type'] = 'short_text'
            
            # SVG properties
            cell['svg_properties'] = self.get_svg_properties(cell)
        
        return cells
    
    def get_svg_properties(self, cell):
        """Get SVG-specific properties for each cell type"""
        
        base_props = {
            'fill': 'none',
            'stroke': '#000000',
            'stroke_width': 1
        }
        
        type_specific = {
            'header': {
                'fill': '#f0f0f0',
                'font_weight': 'bold',
                'font_size': '12px',
                'text_anchor': 'middle'
            },
            'question_number': {
                'font_weight': 'bold',
                'font_size': '11px',
                'text_anchor': 'start'
            },
            'question_text': {
                'font_size': '10px',
                'text_anchor': 'start',
                'word_wrap': True
            },
            'score_value': {
                'font_size': '10px',
                'text_anchor': 'middle',
                'font_weight': 'bold'
            },
            'classification_code': {
                'font_size': '9px',
                'text_anchor': 'middle'
            },
            'empty': {
                'fill': 'white'
            }
        }
        
        props = base_props.copy()
        props.update(type_specific.get(cell['cell_type'], {}))
        
        return props
    
    def create_table_svg_structure(self, cells, image_shape):
        """Create comprehensive SVG structure from parsed table"""
        
        svg_structure = {
            'viewport': {
                'width': image_shape[1],
                'height': image_shape[0]
            },
            'table_structure': {
                'total_cells': len(cells),
                'rows': max(cell['row'] for cell in cells) + 1 if cells else 0,
                'cols': max(cell['col'] for cell in cells) + 1 if cells else 0
            },
            'cells': [],
            'cell_groups': {
                'headers': [],
                'questions': [],
                'scores': [],
                'classifications': [],
                'empty': []
            }
        }
        
        for i, cell in enumerate(cells):
            svg_cell = {
                'id': f'cell_{cell["row"]}_{cell["col"]}',
                'grid_position': {'row': cell['row'], 'col': cell['col']},
                'coordinates': cell['coordinates'],
                'dimensions': {'width': cell['width'], 'height': cell['height']},
                'content': {
                    'text': cell['text_content'],
                    'type': cell['cell_type'],
                    'confidence': cell['ocr_confidence']
                },
                'svg_properties': cell['svg_properties']
            }
            
            svg_structure['cells'].append(svg_cell)
            
            # Group by type
            if cell['cell_type'] == 'header':
                svg_structure['cell_groups']['headers'].append(svg_cell)
            elif cell['cell_type'] in ['question_number', 'question_text', 'long_text']:
                svg_structure['cell_groups']['questions'].append(svg_cell)
            elif cell['cell_type'] == 'score_value':
                svg_structure['cell_groups']['scores'].append(svg_cell)
            elif cell['cell_type'] == 'classification_code':
                svg_structure['cell_groups']['classifications'].append(svg_cell)
            elif cell['cell_type'] == 'empty':
                svg_structure['cell_groups']['empty'].append(svg_cell)
        
        return svg_structure
    
    def visualize_table_parsing(self, image, cells, h_lines, v_lines, image_path):
        """Create comprehensive visualization of table parsing results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(24, 16))
        
        # Color map for cell types
        type_colors = {
            'header': '#FF6B6B',
            'question_number': '#4ECDC4',
            'question_text': '#45B7D1',
            'score_value': '#96CEB4',
            'classification_code': '#FFEAA7',
            'long_text': '#DDA0DD',
            'short_text': '#F0E68C',
            'empty': '#FFFFFF'
        }
        
        # Plot 1: Original with detected lines
        axes[0,0].imshow(image)
        
        # Draw detected lines
        for y in h_lines:
            axes[0,0].axhline(y=y, color='red', linewidth=2, alpha=0.7)
        for x in v_lines:
            axes[0,0].axvline(x=x, color='blue', linewidth=2, alpha=0.7)
        
        axes[0,0].set_title(f"Detected Lines (H: {len(h_lines)}, V: {len(v_lines)})")
        axes[0,0].axis('off')
        
        # Plot 2: Cell grid with types
        axes[0,1].imshow(image)
        
        for cell in cells:
            coords = cell['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            color = type_colors.get(cell['cell_type'], '#808080')
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor='black', 
                                   facecolor=color, alpha=0.3)
            axes[0,1].add_patch(rect)
            
            # Add cell type label
            axes[0,1].text(x1+5, y1+10, cell['cell_type'], 
                         fontsize=6, weight='bold', color='black')
        
        axes[0,1].set_title(f"Cell Classification ({len(cells)} cells)")
        axes[0,1].axis('off')
        
        # Plot 3: Text content visualization
        axes[1,0].imshow(image)
        
        for cell in cells:
            if cell['has_text']:
                coords = cell['coordinates']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                
                # Show text content (truncated)
                text_preview = cell['text_content'][:15] + "..." if len(cell['text_content']) > 15 else cell['text_content']
                
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=1, edgecolor='green', 
                                       facecolor='lightgreen', alpha=0.2)
                axes[1,0].add_patch(rect)
                
                axes[1,0].text(x1+2, y1+8, text_preview, 
                             fontsize=5, color='darkgreen', weight='bold')
        
        axes[1,0].set_title("Text Content (OCR Results)")
        axes[1,0].axis('off')
        
        # Plot 4: Statistics and legend
        axes[1,1].axis('off')
        
        # Count cell types
        type_counts = {}
        for cell in cells:
            cell_type = cell['cell_type']
            type_counts[cell_type] = type_counts.get(cell_type, 0) + 1
        
        # Create legend and stats
        legend_text = "CELL TYPE STATISTICS\n" + "="*25 + "\n"
        
        y_pos = 0.9
        for cell_type, count in type_counts.items():
            color = type_colors.get(cell_type, '#808080')
            
            # Color patch
            rect = patches.Rectangle((0.05, y_pos-0.02), 0.03, 0.03, 
                                   facecolor=color, edgecolor='black')
            axes[1,1].add_patch(rect)
            
            # Text
            axes[1,1].text(0.12, y_pos, f"{cell_type}: {count}", 
                         fontsize=10, verticalalignment='center')
            
            y_pos -= 0.08
        
        # Additional stats
        total_with_text = sum(1 for cell in cells if cell['has_text'])
        avg_confidence = np.mean([cell['ocr_confidence'] for cell in cells if cell['ocr_confidence'] > 0])
        
        stats_text = f"\nTOTAL CELLS: {len(cells)}\n"
        stats_text += f"WITH TEXT: {total_with_text}\n"
        stats_text += f"AVG OCR CONFIDENCE: {avg_confidence:.1f}%" if not np.isnan(avg_confidence) else "AVG OCR CONFIDENCE: N/A"
        
        axes[1,1].text(0.05, 0.3, stats_text, fontsize=12, 
                     verticalalignment='top', fontfamily='monospace')
        
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].set_title("Statistics & Legend")
        
        plt.tight_layout()
        
        # Save visualization
        base_name = Path(image_path).stem
        output_path = f"{base_name}_table_parsing_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Table parsing visualization saved: {output_path}")
    
    def save_table_parsing_results(self, svg_data, cells, image_path):
        """Save detailed table parsing results"""
        
        base_name = Path(image_path).stem
        
        # Save complete results
        complete_data = {
            'image_path': image_path,
            'parsing_summary': {
                'total_cells': len(cells),
                'cells_with_text': sum(1 for cell in cells if cell['has_text']),
                'cell_types': {}
            },
            'svg_structure': svg_data,
            'detailed_cells': cells
        }
        
        # Count cell types
        for cell in cells:
            cell_type = cell['cell_type']
            complete_data['parsing_summary']['cell_types'][cell_type] = \
                complete_data['parsing_summary']['cell_types'].get(cell_type, 0) + 1
        
        # Save files
        with open(f"{base_name}_table_parsing_complete.json", 'w') as f:
            json.dump(complete_data, f, indent=2, default = str)
        
        with open(f"{base_name}_table_svg_structure.json", 'w') as f:
            json.dump(svg_data, f, indent=2, default = str)
        
        print(f"✅ Table parsing results saved:")
        print(f"   Complete: {base_name}_table_parsing_complete.json")
        print(f"   SVG structure: {base_name}_table_svg_structure.json")
        
        # Print summary
        print(f"\n📊 PARSING SUMMARY:")
        print(f"   Total cells: {len(cells)}")
        print(f"   Grid size: {svg_data['table_structure']['rows']}x{svg_data['table_structure']['cols']}")
        print(f"   Cells with text: {sum(1 for cell in cells if cell['has_text'])}")
        
        print(f"\n📋 CELL TYPES:")
        for cell_type, count in complete_data['parsing_summary']['cell_types'].items():
            print(f"   {cell_type}: {count}")

# Usage
if __name__ == "__main__":
    parser = QuestionPaperTableParser()
    
    # Update with your image path
    image_path = "images/sample1.jpg"
    
    if os.path.exists(image_path):
        cells, svg_structure = parser.parse_question_paper_table(image_path)
        
        print(f"\n🎉 Table parsing complete!")
        print(f"Detected {len(cells)} cells in {svg_structure['table_structure']['rows']}x{svg_structure['table_structure']['cols']} grid")
        
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable")