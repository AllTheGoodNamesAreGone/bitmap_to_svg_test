import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def create_svg_from_header_elements(image_path, header_boundary, header_elements, output_svg_path):
    """
    Create an SVG representation of the document with detected elements
    
    Args:
        image_path: Path to the original document image
        header_boundary: Y-coordinate of the boundary between header and body
        header_elements: Dictionary of elements detected in the header
        output_svg_path: Path to save the SVG file
    """
    # Read the original image to get dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Convert image to base64 for embedding (optional)
    img_base64 = image_to_base64(image_path)
    
    # Start building SVG content
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <defs>
        <style>
            .header-region {{ fill: rgba(255, 255, 0, 0.1); stroke: #ff0000; stroke-width: 2; }}
            .body-region {{ fill: rgba(100, 100, 255, 0.1); stroke: #ff0000; stroke-width: 2; }}
            .text-box {{ fill: none; stroke: #00ff00; stroke-width: 2; }}
            .logo-box {{ fill: none; stroke: #0000ff; stroke-width: 3; }}
            .table-box {{ fill: none; stroke: #ff0000; stroke-width: 2; }}
            .boundary-line {{ stroke: #ff0000; stroke-width: 3; }}
            .label-text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #000000; }}
            .region-label {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #ff0000; }}
        </style>
    </defs>
    
    <!-- Background image (optional) -->
    <image x="0" y="0" width="{width}" height="{height}" xlink:href="data:image/jpeg;base64,{img_base64}" opacity="0.8"/>
    
    <!-- Header region -->
    <rect x="0" y="0" width="{width}" height="{header_boundary}" class="header-region"/>
    
    <!-- Body region -->
    <rect x="0" y="{header_boundary}" width="{width}" height="{height - header_boundary}" class="body-region"/>
    
    <!-- Boundary line -->
    <line x1="0" y1="{header_boundary}" x2="{width}" y2="{header_boundary}" class="boundary-line"/>
    
    <!-- Region labels -->
    <text x="10" y="25" class="region-label">HEADER REGION</text>
    <text x="10" y="{header_boundary + 30}" class="region-label">BODY REGION</text>
'''
    
    # Add text elements
    for i, (x, y, w, h, text) in enumerate(header_elements["text"]):
        # Clean text for SVG
        safe_text = escape_xml_text(text)
        truncated_text = safe_text[:20] + ("..." if len(safe_text) > 20 else "")
        
        # Add bounding box
        svg_content += f'''    <!-- Text element {i+1} -->
    <rect x="{x}" y="{y}" width="{w}" height="{h}" class="text-box"/>
    <text x="{x}" y="{max(y - 5, 15)}" class="label-text">TEXT: {truncated_text}</text>
'''
    
    # Add logo elements
    for i, (x, y, w, h) in enumerate(header_elements["logos"]):
        svg_content += f'''    <!-- Logo element {i+1} -->
    <rect x="{x}" y="{y}" width="{w}" height="{h}" class="logo-box"/>
    <text x="{x}" y="{max(y - 5, 15)}" class="label-text">LOGO {i+1}</text>
'''
    
    # Add table elements
    for i, (x, y, w, h) in enumerate(header_elements["tables"]):
        svg_content += f'''    <!-- Table element {i+1} -->
    <rect x="{x}" y="{y}" width="{w}" height="{h}" class="table-box"/>
    <text x="{x}" y="{max(y - 5, 15)}" class="label-text">TABLE {i+1}</text>
'''
    
    # Close SVG
    svg_content += '</svg>'
    
    # Write SVG file
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"SVG file created at: {output_svg_path}")

def image_to_base64(image_path):
    """
    Convert image to base64 string for embedding in SVG
    """
    try:
        # Read image with PIL to ensure proper format
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def escape_xml_text(text):
    """
    Escape special XML characters in text
    """
    if not text:
        return ""
    
    # Replace XML special characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    
    # Remove non-printable characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    return text

def create_structured_svg_with_groups(image_path, header_boundary, header_elements, output_svg_path):
    """
    Create a more structured SVG with grouped elements and better organization
    """
    # Read the original image to get dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Start building SVG content with better structure
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    
    <defs>
        <style><![CDATA[
            .header-region {{ fill: rgba(255, 255, 0, 0.1); stroke: #ff0000; stroke-width: 2; }}
            .body-region {{ fill: rgba(100, 100, 255, 0.1); stroke: #ff0000; stroke-width: 2; }}
            .text-box {{ fill: none; stroke: #00ff00; stroke-width: 2; }}
            .logo-box {{ fill: none; stroke: #0000ff; stroke-width: 3; }}
            .table-box {{ fill: none; stroke: #ff0000; stroke-width: 2; }}
            .boundary-line {{ stroke: #ff0000; stroke-width: 3; }}
            .label-text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #000000; }}
            .region-label {{ font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; fill: #ff0000; }}
            .element-group:hover {{ opacity: 0.8; }}
        ]]></style>
        
        <!-- Patterns for different element types -->
        <pattern id="textPattern" patternUnits="userSpaceOnUse" width="4" height="4">
            <rect width="4" height="4" fill="#e6ffe6"/>
            <circle cx="2" cy="2" r="1" fill="#00ff00"/>
        </pattern>
        
        <pattern id="logoPattern" patternUnits="userSpaceOnUse" width="8" height="8">
            <rect width="8" height="8" fill="#e6e6ff"/>
            <path d="M0,0 L8,8 M8,0 L0,8" stroke="#0000ff" stroke-width="1"/>
        </pattern>
    </defs>
    
    <!-- Document regions group -->
    <g id="document-regions">
        <rect x="0" y="0" width="{width}" height="{header_boundary}" class="header-region"/>
        <rect x="0" y="{header_boundary}" width="{width}" height="{height - header_boundary}" class="body-region"/>
        <line x1="0" y1="{header_boundary}" x2="{width}" y2="{header_boundary}" class="boundary-line"/>
        
        <!-- Region labels -->
        <text x="10" y="25" class="region-label">HEADER REGION</text>
        <text x="10" y="{header_boundary + 30}" class="region-label">BODY REGION</text>
    </g>
    
    <!-- Text elements group -->
    <g id="text-elements">
'''
    
    # Add text elements with metadata
    for i, (x, y, w, h, text) in enumerate(header_elements["text"]):
        safe_text = escape_xml_text(text)
        truncated_text = safe_text[:20] + ("..." if len(safe_text) > 20 else "")
        
        svg_content += f'''        <g class="element-group" id="text-{i+1}">
            <title>Text Element {i+1}: {safe_text}</title>
            <rect x="{x}" y="{y}" width="{w}" height="{h}" class="text-box" fill="url(#textPattern)" fill-opacity="0.3"/>
            <text x="{x}" y="{max(y - 5, 15)}" class="label-text">TEXT: {truncated_text}</text>
        </g>
'''
    
    svg_content += '    </g>\n\n    <!-- Logo elements group -->\n    <g id="logo-elements">\n'
    
    # Add logo elements
    for i, (x, y, w, h) in enumerate(header_elements["logos"]):
        svg_content += f'''        <g class="element-group" id="logo-{i+1}">
            <title>Logo Element {i+1}</title>
            <rect x="{x}" y="{y}" width="{w}" height="{h}" class="logo-box" fill="url(#logoPattern)" fill-opacity="0.3"/>
            <text x="{x}" y="{max(y - 5, 15)}" class="label-text">LOGO {i+1}</text>
        </g>
'''
    
    svg_content += '    </g>\n\n    <!-- Table elements group -->\n    <g id="table-elements">\n'
    
    # Add table elements
    for i, (x, y, w, h) in enumerate(header_elements["tables"]):
        svg_content += f'''        <g class="element-group" id="table-{i+1}">
            <title>Table Element {i+1}</title>
            <rect x="{x}" y="{y}" width="{w}" height="{h}" class="table-box"/>
            <text x="{x}" y="{max(y - 5, 15)}" class="label-text">TABLE {i+1}</text>
        </g>
'''
    
    svg_content += '    </g>\n</svg>'
    
    # Write SVG file
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"Structured SVG file created at: {output_svg_path}")


