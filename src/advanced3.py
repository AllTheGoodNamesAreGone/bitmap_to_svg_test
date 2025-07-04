#!/usr/bin/env python3
"""
Hybrid Detection System for SVG-Ready Question Paper Analysis
Combines multiple detection methods for maximum granularity
"""

import layoutparser as lp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path

class HybridSVGDetectionSystem:
    def __init__(self):
        """Initialize multiple detection models and methods"""
        
        # Model 1: Ultra-low threshold general detection
        self.general_model = lp.Detectron2LayoutModel(
            config_path='models\publaynet_mask_rcnn_R_50_FPN_3x/config.yml',
            model_path='models\publaynet_mask_rcnn_R_50_FPN_3x/model_final.pth',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.05],  # Ultra-low
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        
        # OCR for text-based detection
        try:
            self.ocr_agent = lp.TesseractAgent(languages='eng')
            self.has_ocr = True
        except:
            print("⚠️  OCR not available")
            self.has_ocr = False
        
        print("✅ Hybrid detection system initialized")
    
    def analyze_for_svg_reconstruction(self, image_path):
        """Comprehensive analysis optimized for SVG reconstruction"""
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"🔍 Starting comprehensive SVG-ready analysis of {image_path}")
        
        all_elements = []
        
        # Step 1: Ultra-low threshold detection
        print("📋 Step 1: Ultra-low threshold layout detection...")
        layout_elements = self.ultra_low_threshold_detection(image_rgb)
        all_elements.extend(layout_elements)
        print(f"   Found {len(layout_elements)} layout elements")
        
        # Step 2: Table structure decomposition
        print("📋 Step 2: Table structure decomposition...")
        table_elements = self.decompose_table_structure(image_rgb)
        all_elements.extend(table_elements)
        print(f"   Found {len(table_elements)} table elements")
        
        # Step 3: Text line detection
        print("📋 Step 3: Text line detection...")
        text_elements = self.detect_text_lines(image_rgb)
        all_elements.extend(text_elements)
        print(f"   Found {len(text_elements)} text elements")
        
        # Step 4: Manual grid analysis
        print("📋 Step 4: Manual grid analysis...")
        grid_elements = self.analyze_grid_structure(image_rgb)
        all_elements.extend(grid_elements)
        print(f"   Found {len(grid_elements)} grid elements")
        
        # Step 5: Border and line detection
        print("📋 Step 5: Border and line detection...")
        border_elements = self.detect_borders_and_lines(image_rgb)
        all_elements.extend(border_elements)
        print(f"   Found {len(border_elements)} border/line elements")
        
        # Step 6: Remove duplicates and merge overlapping
        print("📋 Step 6: Processing and deduplication...")
        processed_elements = self.process_and_deduplicate(all_elements, image_rgb.shape)
        
        # Step 7: Create SVG-optimized structure
        svg_structure = self.create_svg_structure(processed_elements, image_rgb.shape)
        
        # Step 8: Visualize comprehensive results
        self.create_comprehensive_visualization(image_rgb, processed_elements, image_path)
        
        # Step 9: Save SVG-ready data
        self.save_svg_ready_data(svg_structure, processed_elements, image_path)
        
        print(f"✅ Analysis complete: {len(processed_elements)} total elements detected")
        return processed_elements, svg_structure
    
    def ultra_low_threshold_detection(self, image):
        """Detect with ultra-low threshold to catch everything"""
        layout_blocks = self.general_model.detect(image)
        
        elements = []
        for i, block in enumerate(layout_blocks):
            elements.append({
                'id': f'layout_{i}',
                'type': block.type,
                'method': 'layoutparser_ultra_low',
                'confidence': float(block.score),
                'coordinates': {
                    'x1': int(block.coordinates[0]),
                    'y1': int(block.coordinates[1]),
                    'x2': int(block.coordinates[2]),
                    'y2': int(block.coordinates[3])
                },
                'svg_type': 'rect' if block.type in ['Table', 'Figure'] else 'text'
            })
        
        return elements
    
    def decompose_table_structure(self, image):
        """Decompose detected tables into individual cells"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find line positions
        h_lines_pos = self.find_line_positions(horizontal_lines, axis=0)  # horizontal
        v_lines_pos = self.find_line_positions(vertical_lines, axis=1)    # vertical
        
        elements = []
        
        # Create table cells from grid intersection
        for i in range(len(h_lines_pos) - 1):
            for j in range(len(v_lines_pos) - 1):
                y1, y2 = h_lines_pos[i], h_lines_pos[i + 1]
                x1, x2 = v_lines_pos[j], v_lines_pos[j + 1]
                
                # Filter out very small cells
                if (x2 - x1) > 30 and (y2 - y1) > 15:
                    elements.append({
                        'id': f'cell_{i}_{j}',
                        'type': 'TableCell',
                        'method': 'manual_table_decomposition',
                        'confidence': 0.9,
                        'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'svg_type': 'rect',
                        'row': i,
                        'col': j
                    })
        
        return elements
    
    def find_line_positions(self, line_image, axis):
        """Find positions of detected lines"""
        if axis == 0:  # horizontal lines
            line_sums = np.sum(line_image, axis=1)
        else:  # vertical lines
            line_sums = np.sum(line_image, axis=0)
        
        # Find peaks (line positions)
        threshold = np.max(line_sums) * 0.3
        lines = []
        
        for i, val in enumerate(line_sums):
            if val > threshold:
                lines.append(i)
        
        # Merge nearby lines
        merged_lines = []
        if lines:
            current_line = lines[0]
            for line in lines[1:]:
                if line - current_line > 10:  # Gap threshold
                    merged_lines.append(current_line)
                    current_line = line
                else:
                    current_line = (current_line + line) // 2
            merged_lines.append(current_line)
        
        return sorted(merged_lines)
    
    def detect_text_lines(self, image):
        """Detect individual text lines for granular SVG text elements"""
        if not self.has_ocr:
            return []
        
        elements = []
        
        try:
            # Use OCR to detect text regions
            import pytesseract
            
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text and confidence > 30:  # Filter low confidence
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i] 
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    elements.append({
                        'id': f'text_{i}',
                        'type': 'TextLine',
                        'method': 'ocr_text_detection',
                        'confidence': confidence / 100.0,
                        'coordinates': {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h},
                        'svg_type': 'text',
                        'text_content': text,
                        'font_size': h * 0.8  # Estimate font size
                    })
                    
        except Exception as e:
            print(f"   ⚠️  OCR text detection failed: {e}")
        
        return elements
    
    def analyze_grid_structure(self, image):
        """Analyze overall grid structure of the document"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect straight lines using HoughLines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=50, maxLineGap=10)
        
        elements = []
        
        if lines is not None:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # Classify as horizontal or vertical
                if abs(y2 - y1) < 10:  # Horizontal line
                    line_type = 'HorizontalLine'
                elif abs(x2 - x1) < 10:  # Vertical line
                    line_type = 'VerticalLine'
                else:
                    line_type = 'DiagonalLine'
                
                elements.append({
                    'id': f'line_{i}',
                    'type': line_type,
                    'method': 'hough_line_detection',
                    'confidence': 0.8,
                    'coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'svg_type': 'line',
                    'stroke_width': 1
                })
        
        return elements
    
    def detect_borders_and_lines(self, image):
        """Detect document borders and structural lines"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect thick borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thick_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours for borders
        contours, _ = cv2.findContours(255 - thick_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        elements = []
        
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio
            if w > 50 and h > 20:
                # Determine if it's a border or content area
                area_ratio = (w * h) / (image.shape[0] * image.shape[1])
                
                if area_ratio > 0.01:  # Significant area
                    elements.append({
                        'id': f'border_{i}',
                        'type': 'Border',
                        'method': 'contour_detection',
                        'confidence': 0.7,
                        'coordinates': {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h},
                        'svg_type': 'rect',
                        'area_ratio': area_ratio
                    })
        
        return elements
    
    def process_and_deduplicate(self, all_elements, image_shape):
        """Remove duplicates and merge overlapping elements"""
        
        def calculate_overlap(elem1, elem2):
            """Calculate overlap ratio between two elements"""
            coords1 = elem1['coordinates']
            coords2 = elem2['coordinates']
            
            # Calculate intersection
            x1 = max(coords1['x1'], coords2['x1'])
            y1 = max(coords1['y1'], coords2['y1'])
            x2 = min(coords1['x2'], coords2['x2'])
            y2 = min(coords1['y2'], coords2['y2'])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (coords1['x2'] - coords1['x1']) * (coords1['y2'] - coords1['y1'])
            area2 = (coords2['x2'] - coords2['x1']) * (coords2['y2'] - coords2['y1'])
            
            return intersection / min(area1, area2)
        
        # Remove duplicates based on overlap
        unique_elements = []
        
        for element in all_elements:
            is_duplicate = False
            
            for existing in unique_elements:
                overlap = calculate_overlap(element, existing)
                
                if overlap > 0.7:  # High overlap threshold
                    # Keep the one with higher confidence
                    if element['confidence'] > existing['confidence']:
                        unique_elements.remove(existing)
                        unique_elements.append(element)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_elements.append(element)
        
        # Sort by position (top to bottom, left to right)
        unique_elements.sort(key=lambda x: (x['coordinates']['y1'], x['coordinates']['x1']))
        
        # Reassign IDs
        for i, element in enumerate(unique_elements):
            element['final_id'] = i
        
        return unique_elements
    
    def create_svg_structure(self, elements, image_shape):
        """Create SVG-optimized structure"""
        
        svg_structure = {
            'viewport': {
                'width': image_shape[1],
                'height': image_shape[0]
            },
            'elements': [],
            'groups': {
                'headers': [],
                'tables': [],
                'text_content': [],
                'borders': [],
                'lines': []
            }
        }
        
        for element in elements:
            svg_element = {
                'id': element['final_id'],
                'type': element['svg_type'],
                'element_type': element['type'],
                'coordinates': element['coordinates'],
                'confidence': element['confidence'],
                'detection_method': element['method']
            }
            
            # Add type-specific properties
            if element['svg_type'] == 'text':
                svg_element.update({
                    'text_content': element.get('text_content', ''),
                    'font_size': element.get('font_size', 12),
                    'font_family': 'serif'
                })
                svg_structure['groups']['text_content'].append(svg_element)
                
            elif element['svg_type'] == 'rect':
                svg_element.update({
                    'fill': 'none',
                    'stroke': '#000000',
                    'stroke_width': 1
                })
                
                if element['type'] in ['Table', 'TableCell']:
                    svg_structure['groups']['tables'].append(svg_element)
                elif element['type'] == 'Border':
                    svg_structure['groups']['borders'].append(svg_element)
                    
            elif element['svg_type'] == 'line':
                svg_element.update({
                    'stroke': '#000000',
                    'stroke_width': element.get('stroke_width', 1)
                })
                svg_structure['groups']['lines'].append(svg_element)
            
            svg_structure['elements'].append(svg_element)
        
        return svg_structure
    
    def create_comprehensive_visualization(self, image, elements, image_path):
        """Create detailed visualization showing all detected elements"""
        
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        
        # Method-based visualization
        method_colors = {
            'layoutparser_ultra_low': '#FF6B6B',
            'manual_table_decomposition': '#4ECDC4', 
            'ocr_text_detection': '#45B7D1',
            'hough_line_detection': '#96CEB4',
            'contour_detection': '#FFEAA7'
        }
        
        # Plot 1: All elements by detection method
        axes[0,0].imshow(image)
        for element in elements:
            coords = element['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            color = method_colors.get(element['method'], '#DDA0DD')
            
            if element['svg_type'] == 'line':
                axes[0,0].plot([x1, x2], [y1, y2], color=color, linewidth=2)
            else:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                axes[0,0].add_patch(rect)
        
        axes[0,0].set_title(f"All Elements by Detection Method ({len(elements)} total)")
        axes[0,0].axis('off')
        
        # Plot 2: Elements by type
        type_colors = {
            'Text': '#2E86AB', 'Title': '#A23B72', 'Table': '#F18F01',
            'TableCell': '#FFA500', 'TextLine': '#87CEEB', 'HorizontalLine': '#32CD32',
            'VerticalLine': '#FF6347', 'Border': '#9370DB'
        }
        
        axes[0,1].imshow(image)
        for element in elements:
            coords = element['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            color = type_colors.get(element['type'], '#808080')
            
            if element['svg_type'] == 'line':
                axes[0,1].plot([x1, x2], [y1, y2], color=color, linewidth=2)
            else:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
                axes[0,1].add_patch(rect)
        
        axes[0,1].set_title("Elements by Type")
        axes[0,1].axis('off')
        
        # Plot 3: Text elements only
        axes[1,0].imshow(image)
        text_elements = [e for e in elements if e['svg_type'] == 'text']
        for element in text_elements:
            coords = element['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
            axes[1,0].add_patch(rect)
            
            # Add text content if available
            text_content = element.get('text_content', '')[:20]
            if text_content:
                axes[1,0].text(x1, y1-5, text_content, fontsize=6, color='blue')
        
        axes[1,0].set_title(f"Text Elements Only ({len(text_elements)} elements)")
        axes[1,0].axis('off')
        
        # Plot 4: Table structure only
        axes[1,1].imshow(image)
        table_elements = [e for e in elements if 'Table' in e['type'] or 'line' in e['svg_type'].lower()]
        for element in table_elements:
            coords = element['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            if element['svg_type'] == 'line':
                axes[1,1].plot([x1, x2], [y1, y2], color='red', linewidth=2)
            else:
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
                axes[1,1].add_patch(rect)
        
        axes[1,1].set_title(f"Table Structure Only ({len(table_elements)} elements)")
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        base_name = Path(image_path).stem
        output_path = f"{base_name}_comprehensive_svg_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive visualization saved: {output_path}")
    
    def save_svg_ready_data(self, svg_structure, elements, image_path):
        """Save SVG-ready data in multiple formats"""
        
        base_name = Path(image_path).stem
        
        # Save detailed analysis
        detailed_data = {
            'image_path': image_path,
            'total_elements': len(elements),
            'svg_structure': svg_structure,
            'detailed_elements': elements,
            'summary': {
                'by_method': {},
                'by_type': {},
                'by_svg_type': {}
            }
        }
        
        # Generate summaries
        for element in elements:
            method = element['method']
            elem_type = element['type']
            svg_type = element['svg_type']
            
            detailed_data['summary']['by_method'][method] = detailed_data['summary']['by_method'].get(method, 0) + 1
            detailed_data['summary']['by_type'][elem_type] = detailed_data['summary']['by_type'].get(elem_type, 0) + 1
            detailed_data['summary']['by_svg_type'][svg_type] = detailed_data['summary']['by_svg_type'].get(svg_type, 0) + 1
        
        # Save complete data
        with open(f"{base_name}_svg_ready_complete.json", 'w') as f:
            json.dump(detailed_data, f, indent=2, default = str)
        
        # Save just SVG structure
        with open(f"{base_name}_svg_structure.json", 'w') as f:
            json.dump(svg_structure, f, indent=2, default = str)
        
        print(f"✅ SVG-ready data saved:")
        print(f"   Complete: {base_name}_svg_ready_complete.json")
        print(f"   SVG structure: {base_name}_svg_structure.json")

# Usage
if __name__ == "__main__":
    analyzer = HybridSVGDetectionSystem()
    
    # Update with your image path
    image_path = "images/sample1.jpg"
    
    if os.path.exists(image_path):
        elements, svg_structure = analyzer.analyze_for_svg_reconstruction(image_path)
        
        print(f"\n🎉 SVG-ready analysis complete!")
        print(f"Total elements detected: {len(elements)}")
        print(f"SVG groups:")
        for group_name, group_elements in svg_structure['groups'].items():
            print(f"  {group_name}: {len(group_elements)} elements")
            
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable")