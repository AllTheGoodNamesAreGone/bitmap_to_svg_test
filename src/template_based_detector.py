import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json
import re
from typing import Dict, List, Tuple, Optional
import base64

class TemplateBasedDetector:
    """Accurate detection using predefined template regions"""
    
    def __init__(self, template_path: str):
        """Initialize with template file"""
        with open(template_path, 'r') as f:
            self.template = json.load(f)
        self.image = None
        self.height = 0
        self.width = 0
        self.detected_elements = {
            "regions": {},
            "text": {},
            "tables": {},
            "logos": {},
            "validation": {}
        }
        
    def load_image(self, image_path: str):
        """Load and prepare image for detection"""
        self.image = cv2.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        print(f"Image loaded: {self.width}x{self.height}")
        
    def percent_to_pixels(self, x_percent: float, y_percent: float, 
                         width_percent: float, height_percent: float) -> Tuple[int, int, int, int]:
        """Convert percentage coordinates to pixel coordinates"""
        x = int(x_percent * self.width / 100)
        y = int(y_percent * self.height / 100)
        w = int(width_percent * self.width / 100)
        h = int(height_percent * self.height / 100)
        return x, y, w, h
        
    def extract_region(self, region_coords: Dict) -> np.ndarray:
        """Extract image region based on coordinates"""
        x, y, w, h = self.percent_to_pixels(
            region_coords['x_percent'],
            region_coords['y_percent'],
            region_coords['width_percent'],
            region_coords['height_percent']
        )
        return self.image[y:y+h, x:x+w]
        
    def detect_logo_in_region(self, region_name: str) -> Dict:
        """Detect logo within specified region with high accuracy"""
        region_config = self.get_region_config(region_name)
        if not region_config:
            return {}
            
        region_img = self.extract_region(region_config)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Multiple logo detection methods for accuracy
        logo_candidates = []
        
        # Method 1: Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 30:  # Minimum logo size
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / (w * h)
                aspect_ratio = w / h
                
                # Logo characteristics: decent edge density, reasonable aspect ratio
                if edge_density > 0.03 and 0.5 < aspect_ratio < 2.0:
                    logo_candidates.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'edge_density': edge_density,
                        'aspect_ratio': aspect_ratio,
                        'confidence': edge_density * 100
                    })
        
        # Method 2: Template matching (if reference logo available)
        # This would be enhanced with actual logo templates
        
        # Method 3: Color analysis for logo detection
        # Logos often have distinct color patterns
        
        # Select best candidate
        if logo_candidates:
            best_logo = max(logo_candidates, key=lambda x: x['confidence'])
            
            # Convert back to full image coordinates
            region_x, region_y, _, _ = self.percent_to_pixels(
                region_config['x_percent'], region_config['y_percent'],
                region_config['width_percent'], region_config['height_percent']
            )
            
            best_logo['x'] += region_x
            best_logo['y'] += region_y
            
            return best_logo
            
        return {}
        
    def detect_text_in_region(self, region_name: str) -> List[Dict]:
        """Detect and extract text within specified region"""
        region_config = self.get_region_config(region_name)
        if not region_config:
            return []
            
        region_img = self.extract_region(region_config)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Enhance text region for better OCR
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Noise reduction
        kernel = np.ones((1,1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # OCR with optimized settings for the region type
        custom_config = self.get_ocr_config(region_name)
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)
        
        text_blocks = []
        region_x, region_y, _, _ = self.percent_to_pixels(
            region_config['x_percent'], region_config['y_percent'],
            region_config['width_percent'], region_config['height_percent']
        )
        
        for i in range(len(data['text'])):
            confidence = int(data['conf'][i])
            text = data['text'][i].strip()
            
            if confidence > 30 and text:  # Confidence threshold
                x = data['left'][i] + region_x
                y = data['top'][i] + region_y
                w = data['width'][i]
                h = data['height'][i]
                
                text_blocks.append({
                    'text': text,
                    'x': x, 'y': y, 'width': w, 'height': h,
                    'confidence': confidence,
                    'region': region_name
                })
                
        return self.merge_text_blocks(text_blocks)
        
    def detect_table_in_region(self, region_name: str) -> Dict:
        """Detect table structure within specified region"""
        region_config = self.get_region_config(region_name)
        if not region_config:
            return {}
            
        region_img = self.extract_region(region_config)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        
        # Table detection using line analysis
        table_info = {
            'region': region_name,
            'cells': [],
            'rows': 0,
            'columns': 0,
            'structure': {}
        }
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Find line intersections for table grid
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Detect table cells using contour analysis
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and sort contours to identify table cells
        cell_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Table cells have reasonable size and aspect ratio
            if area > 500 and 0.1 < aspect_ratio < 10:
                cell_candidates.append((x, y, w, h))
                
        # Extract text from each cell
        region_x, region_y, _, _ = self.percent_to_pixels(
            region_config['x_percent'], region_config['y_percent'],
            region_config['width_percent'], region_config['height_percent']
        )
        
        for x, y, w, h in cell_candidates:
            cell_img = gray[y:y+h, x:x+w]
            cell_text = pytesseract.image_to_string(cell_img, config='--psm 8').strip()
            
            if cell_text:
                table_info['cells'].append({
                    'x': x + region_x,
                    'y': y + region_y,
                    'width': w,
                    'height': h,
                    'text': cell_text
                })
        
        # Organize cells into grid structure
        table_info['structure'] = self.organize_table_cells(table_info['cells'])
        
        return table_info
        
    def get_region_config(self, region_name: str) -> Optional[Dict]:
        """Get configuration for a specific region"""
        # Search through template hierarchy
        layout = self.template.get('layout', {})
        
        # Check header regions
        header_sections = layout.get('header', {}).get('sections', {})
        for section_name, section_data in header_sections.items():
            if 'regions' in section_data:
                if region_name in section_data['regions']:
                    return section_data['regions'][region_name]
                    
        # Check body regions
        body_sections = layout.get('body', {}).get('sections', {})
        for section_name, section_data in body_sections.items():
            if 'regions' in section_data:
                if region_name in section_data['regions']:
                    return section_data['regions'][region_name]
                    
        return None
        
    def get_ocr_config(self, region_name: str) -> str:
        """Get optimized OCR configuration based on region type"""
        region_config = self.get_region_config(region_name)
        if not region_config:
            return '--oem 3 --psm 6'
            
        region_type = region_config.get('type', 'text')
        
        # Optimized OCR settings for different region types
        configs = {
            'text': '--oem 3 --psm 6',  # Uniform block of text
            'table': '--oem 3 --psm 6',  # Uniform block
            'logo': '--oem 3 --psm 8',  # Single word
            'icon_text': '--oem 3 --psm 7',  # Single text line
            'single_line': '--oem 3 --psm 7'  # Single text line
        }
        
        return configs.get(region_type, '--oem 3 --psm 6')
        
    def merge_text_blocks(self, text_blocks: List[Dict]) -> List[Dict]:
        """Merge nearby text blocks for better readability"""
        if not text_blocks:
            return []
            
        # Sort by y-coordinate
        text_blocks.sort(key=lambda x: x['y'])
        
        merged = []
        current_group = [text_blocks[0]]
        
        for block in text_blocks[1:]:
            last_block = current_group[-1]
            
            # Check if blocks should be merged (same line or close proximity)
            vertical_gap = abs(block['y'] - (last_block['y'] + last_block['height']))
            horizontal_gap = abs(block['x'] - (last_block['x'] + last_block['width']))
            
            if vertical_gap < 15 and horizontal_gap < 30:
                current_group.append(block)
            else:
                merged.append(self.create_merged_block(current_group))
                current_group = [block]
                
        merged.append(self.create_merged_block(current_group))
        return merged
        
    def create_merged_block(self, blocks: List[Dict]) -> Dict:
        """Create single block from multiple text blocks"""
        if len(blocks) == 1:
            return blocks[0]
            
        min_x = min(b['x'] for b in blocks)
        min_y = min(b['y'] for b in blocks)
        max_x = max(b['x'] + b['width'] for b in blocks)
        max_y = max(b['y'] + b['height'] for b in blocks)
        
        merged_text = ' '.join(b['text'] for b in blocks)
        avg_confidence = sum(b['confidence'] for b in blocks) / len(blocks)
        
        return {
            'text': merged_text,
            'x': min_x, 'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'confidence': avg_confidence,
            'region': blocks[0]['region']
        }
        
    def organize_table_cells(self, cells: List[Dict]) -> Dict:
        """Organize detected cells into table structure"""
        if not cells:
            return {}
            
        # Sort cells by position (top-to-bottom, left-to-right)
        cells.sort(key=lambda c: (c['y'], c['x']))
        
        # Group cells into rows
        rows = []
        current_row = [cells[0]]
        
        for cell in cells[1:]:
            # Check if cell is in same row (similar y-coordinate)
            if abs(cell['y'] - current_row[0]['y']) < 20:
                current_row.append(cell)
            else:
                # Sort current row by x-coordinate
                current_row.sort(key=lambda c: c['x'])
                rows.append(current_row)
                current_row = [cell]
                
        # Add last row
        current_row.sort(key=lambda c: c['x'])
        rows.append(current_row)
        
        return {
            'rows': len(rows),
            'columns': max(len(row) for row in rows) if rows else 0,
            'grid': rows
        }
        
    def detect_all_regions(self) -> Dict:
        """Detect all elements using template-based approach"""
        if self.image is None:
            raise ValueError("No image loaded. Call load_image() first.")
            
        results = {
            'logos': [],
            'text_regions': [],
            'tables': [],
            'validation': {}
        }
        
        # Detect logos
        logo_regions = ['logo']
        for region in logo_regions:
            logo_data = self.detect_logo_in_region(region)
            if logo_data:
                results['logos'].append(logo_data)
                
        # Detect text in all text regions
        text_regions = ['department_info', 'program_info', 'candidate_instructions', 'marks_info']
        for region in text_regions:
            text_data = self.detect_text_in_region(region)
            if text_data:
                results['text_regions'].extend(text_data)
                
        # Detect tables
        table_regions = ['course_table', 'main_table']
        for region in table_regions:
            table_data = self.detect_table_in_region(region)
            if table_data and table_data.get('cells'):
                results['tables'].append(table_data)
                
        # Validate results against template requirements
        results['validation'] = self.validate_detection_results(results)
        
        return results
        
    def validate_detection_results(self, results: Dict) -> Dict:
        """Validate detection results against template rules"""
        validation = {
            'required_regions_found': True,
            'missing_regions': [],
            'confidence_scores': {},
            'pattern_matches': {}
        }
        
        # Check required regions
        required_regions = self.template.get('validation_rules', {}).get('required_regions', [])
        found_regions = set()
        
        # Collect found regions
        for text_block in results['text_regions']:
            found_regions.add(text_block['region'])
            
        for logo in results['logos']:
            found_regions.add('logo')
            
        for table in results['tables']:
            found_regions.add(table['region'])
            
        # Check for missing regions
        missing = set(required_regions) - found_regions
        if missing:
            validation['required_regions_found'] = False
            validation['missing_regions'] = list(missing)
            
        # Validate text patterns
        text_patterns = self.template.get('validation_rules', {}).get('text_patterns', {})
        for region, pattern in text_patterns.items():
            region_texts = [tb['text'] for tb in results['text_regions'] if tb['region'] == region]
            if region_texts:
                combined_text = ' '.join(region_texts)
                matches = bool(re.search(pattern, combined_text, re.IGNORECASE))
                validation['pattern_matches'][region] = matches
                
        return validation
        
    def generate_svg_with_detections(self, output_path: str, results: Dict):
        """Generate SVG with all detected elements marked"""
        # Convert image to base64 for embedding
        _, buffer = cv2.imencode('.jpg', self.image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
<style>
    .logo-box {{ fill: rgba(255, 0, 0, 0.2); stroke: #FF0000; stroke-width: 2; }}
    .text-box {{ fill: rgba(0, 255, 0, 0.2); stroke: #00FF00; stroke-width: 1; }}
    .table-box {{ fill: rgba(0, 0, 255, 0.2); stroke: #0000FF; stroke-width: 1; }}
    .cell-box {{ fill: none; stroke: #666666; stroke-width: 0.5; }}
    .label {{ font-family: Arial; font-size: 10px; fill: #000; }}
    .template-region {{ fill: rgba(255, 255, 0, 0.1); stroke: #FFAA00; stroke-width: 1; stroke-dasharray: 3,3; }}
</style>

<!-- Background image -->
<image href="data:image/jpeg;base64,{img_base64}" width="{self.width}" height="{self.height}" />

<!-- Template regions (for reference) -->
'''
        
        # Add template regions as reference
        all_regions = self.get_all_template_regions()
        for region_name, coords in all_regions.items():
            x, y, w, h = self.percent_to_pixels(
                coords['x_percent'], coords['y_percent'],
                coords['width_percent'], coords['height_percent']
            )
            svg_content += f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="template-region" />\n'
            svg_content += f'<text x="{x+2}" y="{y-2}" class="label">{region_name}</text>\n'
        
        # Add detected logos
        for logo in results['logos']:
            svg_content += f'''<rect x="{logo['x']}" y="{logo['y']}" width="{logo['width']}" height="{logo['height']}" class="logo-box" />
<text x="{logo['x']+2}" y="{logo['y']-2}" class="label">LOGO ({logo['confidence']:.1f}%)</text>
'''
        
        # Add detected text regions
        for text_block in results['text_regions']:
            safe_text = text_block['text'][:30].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            svg_content += f'''<rect x="{text_block['x']}" y="{text_block['y']}" width="{text_block['width']}" height="{text_block['height']}" class="text-box" />
<text x="{text_block['x']+2}" y="{text_block['y']-2}" class="label">{safe_text}</text>
'''
        
        # Add detected tables
        for table in results['tables']:
            # Table outline
            if table['cells']:
                min_x = min(cell['x'] for cell in table['cells'])
                min_y = min(cell['y'] for cell in table['cells'])
                max_x = max(cell['x'] + cell['width'] for cell in table['cells'])
                max_y = max(cell['y'] + cell['height'] for cell in table['cells'])
                
                svg_content += f'<rect x="{min_x}" y="{min_y}" width="{max_x-min_x}" height="{max_y-min_y}" class="table-box" />\n'
                
                # Individual cells
                for cell in table['cells']:
                    svg_content += f'<rect x="{cell["x"]}" y="{cell["y"]}" width="{cell["width"]}" height="{cell["height"]}" class="cell-box" />\n'
        
        svg_content += '</svg>'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
    def get_all_template_regions(self) -> Dict:
        """Get all regions defined in template"""
        regions = {}
        layout = self.template.get('layout', {})
        
        # Collect all regions from header and body
        for section_type in ['header', 'body']:
            sections = layout.get(section_type, {}).get('sections', {})
            for section_name, section_data in sections.items():
                if 'regions' in section_data:
                    regions.update(section_data['regions'])
                    
        return regions

# Usage example
def main():
    # Load template and image
    detector = TemplateBasedDetector('ramaiah_template.json')
    detector.load_image('question_paper.jpg')
    
    # Detect all elements
    results = detector.detect_all_regions()
    
    # Print results
    print("Detection Results:")
    print(f"Logos found: {len(results['logos'])}")
    print(f"Text regions found: {len(results['text_regions'])}")
    print(f"Tables found: {len(results['tables'])}")
    print(f"Validation: {results['validation']}")
    
    # Generate SVG with detections
    detector.generate_svg_with_detections('detected_elements.svg', results)
    print("SVG generated with detected elements")

if __name__ == "__main__":
    main()