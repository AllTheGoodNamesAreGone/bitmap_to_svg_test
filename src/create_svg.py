#!/usr/bin/env python3
"""
Fixed SVG Creator with proper type handling
Converts string coordinates to integers
"""

import json
import cv2
import numpy as np
from pathlib import Path
import pytesseract
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def safe_int(value, default=0):
    """Safely convert value to integer"""
    try:
        if isinstance(value, str):
            return int(float(value))
        return int(value)
    except (ValueError, TypeError):
        return default

class FixedSVGCreator:
    def __init__(self):
        """Initialize SVG creator with basic templates"""
        
        self.svg_styles = {
            'text': {
                'font-family': 'Times, serif',
                'font-size': '12px',
                'fill': '#000000'
            },
            'header': {
                'font-family': 'Times, serif',
                'font-size': '14px',
                'font-weight': 'bold',
                'fill': '#000000'
            },
            'title': {
                'font-family': 'Times, serif', 
                'font-size': '16px',
                'font-weight': 'bold',
                'fill': '#000000',
                'text-anchor': 'middle'
            }
        }
    
    def load_detection_results(self, json_file):
        """Load detection results from hybrid analysis"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Try different possible structures
            if 'detailed_elements' in data:
                elements = data['detailed_elements']
            elif 'svg_structure' in data and 'elements' in data['svg_structure']:
                elements = data['svg_structure']['elements']
            elif 'strategies' in data:
                # From hybrid analysis - get the best strategy
                best_strategy = None
                max_count = 0
                for strategy_name, strategy_data in data['strategies'].items():
                    if strategy_data['count'] > max_count:
                        max_count = strategy_data['count']
                        best_strategy = strategy_name
                
                if best_strategy:
                    elements = data['strategies'][best_strategy]['blocks']
                else:
                    elements = []
            else:
                # Try direct elements list
                elements = data if isinstance(data, list) else []
            
            print(f"✅ Loaded {len(elements)} elements from {json_file}")
            return self.normalize_coordinates(elements)
            
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
            return []
    
    def normalize_coordinates(self, elements):
        """Normalize coordinate formats and convert to integers"""
        
        normalized_elements = []
        
        for element in elements:
            try:
                # Make a copy to avoid modifying original
                norm_element = element.copy()
                
                # Handle different coordinate formats
                if 'coordinates' in element:
                    coords = element['coordinates']
                    
                    if isinstance(coords, dict):
                        # Dictionary format: {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
                        x1 = safe_int(coords.get('x1', 0))
                        y1 = safe_int(coords.get('y1', 0))
                        x2 = safe_int(coords.get('x2', 0))
                        y2 = safe_int(coords.get('y2', 0))
                    elif isinstance(coords, list) and len(coords) >= 4:
                        # List format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = [safe_int(c) for c in coords[:4]]
                    else:
                        print(f"⚠️  Skipping element with invalid coordinates: {coords}")
                        continue
                    
                    # Ensure coordinates are valid
                    if x2 <= x1 or y2 <= y1:
                        print(f"⚠️  Skipping element with invalid bounds: {x1},{y1},{x2},{y2}")
                        continue
                    
                    # Store normalized coordinates
                    norm_element['coordinates'] = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    }
                    norm_element['width'] = x2 - x1
                    norm_element['height'] = y2 - y1
                    norm_element['area'] = (x2 - x1) * (y2 - y1)
                    
                else:
                    print(f"⚠️  Skipping element without coordinates")
                    continue
                
                normalized_elements.append(norm_element)
                
            except Exception as e:
                print(f"⚠️  Error normalizing element: {e}")
                continue
        
        print(f"✅ Normalized {len(normalized_elements)} elements")
        return normalized_elements
    
    def extract_text_from_elements(self, image_path, elements):
        """Extract text content from detected elements using OCR"""
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return elements
        
        print("🔍 Extracting text from elements...")
        
        text_extracted = 0
        
        for i, element in enumerate(elements):
            try:
                coords = element['coordinates']
                x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                
                # Skip very small regions
                if (x2 - x1) < 15 or (y2 - y1) < 8:
                    element['text_content'] = ''
                    element['has_text'] = False
                    continue
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, image.shape[1]))
                y1 = max(0, min(y1, image.shape[0]))
                x2 = max(x1, min(x2, image.shape[1]))
                y2 = max(y1, min(y2, image.shape[0]))
                
                if x2 <= x1 or y2 <= y1:
                    element['text_content'] = ''
                    element['has_text'] = False
                    continue
                
                # Extract region
                region = image[y1:y2, x1:x2]
                
                if region.size == 0:
                    element['text_content'] = ''
                    element['has_text'] = False
                    continue
                
                # OCR extraction
                text = pytesseract.image_to_string(region, config='--psm 6').strip()
                
                # Clean text
                text = ' '.join(text.split())  # Remove extra whitespace
                
                # Store text content
                element['text_content'] = text
                element['has_text'] = len(text) > 0
                
                if element['has_text']:
                    text_extracted += 1
                
                if i % 100 == 0:
                    print(f"   Processed {i}/{len(elements)} elements...")
                    
            except Exception as e:
                element['text_content'] = ''
                element['has_text'] = False
                continue
        
        print(f"✅ Extracted text from {text_extracted} elements")
        return elements
    
    def classify_elements_simple(self, elements):
        """Simple classification of elements based on content and position"""
        
        print("🏷️  Classifying elements...")
        
        for element in elements:
            text = element.get('text_content', '').lower().strip()
            coords = element['coordinates']
            
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            width = element['width']
            height = element['height']
            
            # Classification logic
            element_type = 'unknown'
            svg_type = 'rect'
            
            if not element.get('has_text', False):
                if width > 100 or height > 100:
                    element_type = 'border'
                else:
                    element_type = 'line'
                svg_type = 'rect'
                
            elif y1 < 150:  # Top of page
                if any(word in text for word in ['ramaiah', 'institute', 'technology', 'dept']):
                    element_type = 'header'
                    svg_type = 'text'
                elif any(word in text for word in ['programme', 'assessment', 'internal', 'computer science']):
                    element_type = 'title'
                    svg_type = 'text'
                elif any(word in text for word in ['term:', 'date:', 'course name:', 'course code:']):
                    element_type = 'course_info'
                    svg_type = 'text'
                    
            elif any(word in text for word in ['instructions', 'answer', 'candidates']):
                element_type = 'instructions'
                svg_type = 'text'
                
            elif any(word in text for word in ['q.no', 'questions', 'blooms', 'levels', 'marks', 'co']):
                element_type = 'table_header'
                svg_type = 'text'
                
            elif text.startswith(('1.', '2.', '3.', '1a', '1b', '2a', '2b')) or text in ['1a', '1b', '1c', '2a', '2b', '2c', '3a']:
                element_type = 'question_number'
                svg_type = 'text'
                
            elif any(word in text for word in ['discuss', 'explain', 'describe', 'differentiate', 'derive', 'what is', 'calculate']):
                element_type = 'question_text'
                svg_type = 'text'
                
            elif text.isdigit() and len(text) <= 3:
                element_type = 'marks'
                svg_type = 'text'
                
            elif text.lower() in ['l1', 'l2', 'l3', 'co1', 'co2', 'co3', 'co4', 'co5']:
                element_type = 'classification'
                svg_type = 'text'
                
            elif width < 150 and height < 40 and len(text) < 20:
                element_type = 'table_cell'
                svg_type = 'text'
                
            elif len(text) > 30:
                element_type = 'text_block'
                svg_type = 'text'
                
            else:
                element_type = 'other_text'
                svg_type = 'text'
            
            element['classified_type'] = element_type
            element['svg_type'] = svg_type
        
        # Print classification summary
        type_counts = {}
        for element in elements:
            elem_type = element.get('classified_type', 'unknown')
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        
        print("📊 Classification summary:")
        for elem_type, count in sorted(type_counts.items()):
            print(f"   {elem_type}: {count}")
        
        return elements
    
    def create_svg_structure(self, elements, image_shape):
        """Create SVG structure from classified elements"""
        
        print("🎨 Creating SVG structure...")
        
        # Create root SVG element
        svg = Element('svg')
        svg.set('width', str(image_shape[1]))
        svg.set('height', str(image_shape[0]))
        svg.set('xmlns', 'http://www.w3.org/2000/svg')
        svg.set('viewBox', f'0 0 {image_shape[1]} {image_shape[0]}')
        
        # Add styles
        style_element = SubElement(svg, 'style')
        style_content = """
        .header { font-family: Times, serif; font-size: 14px; font-weight: bold; fill: #000; }
        .title { font-family: Times, serif; font-size: 16px; font-weight: bold; fill: #000; }
        .course-info { font-family: Times, serif; font-size: 11px; fill: #000; }
        .question-number { font-family: Times, serif; font-size: 12px; font-weight: bold; fill: #000; }
        .question-text { font-family: Times, serif; font-size: 11px; fill: #000; }
        .table-header { font-family: Times, serif; font-size: 10px; font-weight: bold; fill: #000; text-anchor: middle; }
        .table-cell { font-family: Times, serif; font-size: 10px; fill: #000; text-anchor: middle; }
        .instructions { font-family: Times, serif; font-size: 12px; font-style: italic; fill: #000; }
        .classification { font-family: Times, serif; font-size: 9px; fill: #000; text-anchor: middle; }
        .marks { font-family: Times, serif; font-size: 10px; font-weight: bold; fill: #000; text-anchor: middle; }
        .text-block { font-family: Times, serif; font-size: 11px; fill: #000; }
        .other-text { font-family: Times, serif; font-size: 10px; fill: #000; }
        .border { fill: none; stroke: #000; stroke-width: 1; }
        .line { fill: none; stroke: #000; stroke-width: 0.5; }
        """
        style_element.text = style_content
        
        # Sort elements by type for better organization
        text_elements = [e for e in elements if e.get('svg_type') == 'text' and e.get('has_text', False)]
        border_elements = [e for e in elements if e.get('svg_type') == 'rect']
        
        # Add border elements first (so they appear behind text)
        if border_elements:
            border_group = SubElement(svg, 'g', id='borders')
            for i, element in enumerate(border_elements):
                coords = element['coordinates']
                x, y = coords['x1'], coords['y1']
                width, height = element['width'], element['height']
                
                rect = SubElement(border_group, 'rect')
                rect.set('x', str(x))
                rect.set('y', str(y))
                rect.set('width', str(width))
                rect.set('height', str(height))
                rect.set('class', element.get('classified_type', 'border'))
                rect.set('id', f'border_{i}')
        
        # Group text elements by type
        type_groups = {}
        for element in text_elements:
            elem_type = element.get('classified_type', 'other')
            if elem_type not in type_groups:
                type_groups[elem_type] = []
            type_groups[elem_type].append(element)
        
        # Create groups for each element type
        for elem_type, elements_of_type in type_groups.items():
            if not elements_of_type:
                continue
                
            group = SubElement(svg, 'g', id=f'{elem_type}_group')
            
            for i, element in enumerate(elements_of_type):
                coords = element['coordinates']
                x, y = coords['x1'], coords['y1']
                
                text_content = element.get('text_content', '').strip()
                if not text_content:
                    continue
                
                # Create text element
                text_elem = SubElement(group, 'text')
                text_elem.set('x', str(x + 3))  # Small offset from border
                text_elem.set('y', str(y + 12))  # Offset to position text properly
                text_elem.set('class', elem_type.replace('_', '-'))
                text_elem.set('id', f'{elem_type}_{i}')
                
                # Handle long text by breaking it
                if len(text_content) > 80:
                    # Create tspan for text wrapping
                    words = text_content.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        if len(' '.join(current_line + [word])) <= 80:
                            current_line.append(word)
                        else:
                            if current_line:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                            else:
                                lines.append(word)
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    # Add tspan elements for each line
                    for j, line in enumerate(lines[:3]):  # Limit to 3 lines
                        tspan = SubElement(text_elem, 'tspan')
                        tspan.set('x', str(x + 3))
                        tspan.set('dy', '12' if j > 0 else '0')
                        tspan.text = line
                else:
                    text_elem.text = text_content
        
        return svg
    
    def save_svg(self, svg_element, output_path):
        """Save SVG to file with proper formatting"""
        
        # Convert to string and format
        rough_string = tostring(svg_element, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_svg = reparsed.toprettyxml(indent="  ")
        
        # Remove empty lines and fix formatting
        lines = [line for line in pretty_svg.split('\n') if line.strip()]
        formatted_svg = '\n'.join(lines)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_svg)
        
        print(f"✅ SVG saved to: {output_path}")
        return output_path
    
    def create_svg_from_detection(self, json_file, image_path, output_path=None):
        """Complete pipeline: load detection results and create SVG"""
        
        print("🚀 Starting SVG creation pipeline...")
        
        # Load detection results
        elements = self.load_detection_results(json_file)
        if not elements:
            return None
        
        # Extract text content
        elements = self.extract_text_from_elements(image_path, elements)
        
        # Classify elements
        elements = self.classify_elements_simple(elements)
        
        # Get image dimensions
        image = cv2.imread(image_path)
        image_shape = image.shape
        
        # Create SVG structure
        svg_element = self.create_svg_structure(elements, image_shape)
        
        # Save SVG
        if output_path is None:
            base_name = Path(image_path).stem
            output_path = f"{base_name}_reconstructed.svg"
        
        svg_path = self.save_svg(svg_element, output_path)
        
        # Save processed data for review
        processed_data = {
            'total_elements': len(elements),
            'classified_elements': elements,
            'svg_path': str(svg_path)
        }
        
        data_path = f"{Path(output_path).stem}_data.json"
        with open(data_path, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        print(f"✅ Processed data saved to: {data_path}")
        return svg_path, processed_data
    
    def create_preview_html(self, svg_path):
        """Create HTML preview of the SVG"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Question Paper SVG Preview</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .svg-container {{ 
            border: 1px solid #ccc; 
            padding: 20px; 
            background: white; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: auto;
        }}
        h1 {{ color: #333; }}
        .info {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Question Paper SVG Reconstruction</h1>
        
        <div class="info">
            <p><strong>Generated from:</strong> Hybrid detection system with {1294} elements</p>
            <p><strong>SVG file:</strong> {Path(svg_path).name}</p>
            <p><strong>Note:</strong> This is a simple reconstruction. Elements are classified and positioned based on detected coordinates.</p>
        </div>
        
        <div class="svg-container">
            {open(svg_path, 'r', encoding='utf-8').read()}
        </div>
        
        <div class="info">
            <h3>Instructions:</h3>
            <ul>
                <li>The SVG preserves the original layout structure</li>
                <li>Text elements are classified (header, title, question, etc.)</li>
                <li>You can edit the SVG file directly to improve positioning</li>
                <li>Use this as a starting point for fine-tuning detection</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        html_path = f"{Path(svg_path).stem}_preview.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML preview created: {html_path}")
        return html_path

# Usage function
def run_fixed_svg_creation():
    """Run the complete fixed SVG creation pipeline"""
    
    creator = FixedSVGCreator()
    
    # Try to find the JSON file automatically
    json_files = list(Path('.').glob("*svg*.json")) + list(Path('.').glob("*sample*.json"))
    
    if not json_files:
        print("❌ No JSON files found. Looking for files...")
        all_json = list(Path('.').glob("*.json"))
        if all_json:
            print("Available JSON files:")
            for f in all_json:
                print(f"  • {f}")
        return None
    
    json_file = json_files[0]  # Use first found file
    print(f"📄 Using JSON file: {json_file}")
    
    # Try to find image file
    image_files = (list(Path('.').glob("*.jpg")) + 
                  list(Path('.').glob("*.png")) + 
                  list(Path('images').glob("*.jpg")) + 
                  list(Path('images').glob("*.png")))
    
    if not image_files:
        print("❌ No image files found")
        return None
    
    image_path = image_files[0]  # Use first found image
    print(f"🖼️  Using image file: {image_path}")
    
    try:
        # Create SVG
        svg_path, data = creator.create_svg_from_detection("sample1_svg_ready_complete.json", "images/sample1.jpg")
        
        if svg_path:
            # Create HTML preview
            html_path = creator.create_preview_html(svg_path)
            
            print(f"\n🎉 SVG Creation Complete!")
            print(f"📄 SVG file: {svg_path}")
            print(f"🌐 HTML preview: {html_path}")
            
            # Summary
            type_counts = {}
            for element in data['classified_elements']:
                elem_type = element.get('classified_type', 'unknown')
                type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
            
            print(f"\n📊 Classification Results:")
            for elem_type, count in sorted(type_counts.items()):
                print(f"   {elem_type}: {count}")
            
            print(f"\n🚀 Next steps:")
            print(f"1. Open {html_path} in your browser")
            print(f"2. Review the SVG reconstruction quality")
            print(f"3. Check if text positioning is accurate")
            print(f"4. Identify any classification errors")
            
            return svg_path
        else:
            print("❌ SVG creation failed")
            return None
            
    except Exception as e:
        print(f"❌ Error during SVG creation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_fixed_svg_creation()