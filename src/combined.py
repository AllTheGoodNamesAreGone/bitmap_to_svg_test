import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
import re
import base64
from io import BytesIO
from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, fallback_header_detection

def process_question_paper_to_svg(image_path, output_svg_path):
    """
    Process a question paper image to SVG with header and body regions properly marked
    
    Args:
        image_path: Path to the question paper image
        output_svg_path: Path to save the output SVG file
    """
    # Step 1: Split the image into header and body using your existing code
    header_img, body_img, header_boundary = detect_header_with_instructions_and_show_boxes(
        image_path, 
        header_path='temp_header.jpg', 
        body_path='temp_body.jpg', 
        split_image_path='temp_split'
    )
    
    # Step 2: Process the header using document layout analysis
    header_elements = analyze_header_layout(header_img)
    
    # Step 3: Convert the entire document to SVG with proper regions marked
    create_document_svg(
        image_path, 
        header_boundary, 
        header_elements, 
        output_svg_path
    )
    
    # Clean up temporary files
    for temp_file in ['temp_header.jpg', 'temp_body.jpg', 'temp_split/temp_boundary.jpg']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"SVG generated successfully at {output_svg_path}")

def analyze_header_layout(header_img):
    """
    Analyze the header layout to detect text areas, logos, and other elements
    
    Args:
        header_img: Header image
    
    Returns:
        Dictionary containing detected elements in the header
    """
    # Convert to grayscale
    gray = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Create a dictionary to store detected elements
    detected_elements = {
        "text": [],
        "logos": [],
        "tables": []
    }
    
    # Detect text areas in the header using OCR
    custom_config = r'--oem 3 --psm 1'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)
    
    # Process text regions
    text_blocks = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            text = data['text'][i]
            text_blocks.append((x, y, w, h, text))
    
    # Merge close text blocks into logical sections
    merged_text_blocks = merge_text_blocks(text_blocks)
    detected_elements["text"] = merged_text_blocks
    
    # Detect logo using the same method from your first code
    logo_boxes = detect_logo(gray, height, width)
    detected_elements["logos"] = logo_boxes
    
    # Detect tables in the header (if any)
    table_boxes = detect_tables(gray, height, width)
    detected_elements["tables"] = table_boxes
    
    return detected_elements

def merge_text_blocks(text_blocks, horizontal_threshold=20, vertical_threshold=10):
    """
    Merge close text blocks into logical sections
    
    Args:
        text_blocks: List of tuples (x, y, w, h, text)
        horizontal_threshold: Maximum horizontal gap to consider blocks in the same line
        vertical_threshold: Maximum vertical gap to consider blocks in the same paragraph
    
    Returns:
        List of merged text blocks as (x, y, w, h, text)
    """
    if not text_blocks:
        return []
    
    # Sort by y-coordinate
    text_blocks.sort(key=lambda block: block[1])
    
    # Group blocks by horizontal and vertical proximity
    grouped_blocks = []
    current_group = [text_blocks[0]]
    
    for block in text_blocks[1:]:
        prev_block = current_group[-1]
        
        # Check if current block is on the same line or close to previous block
        if (abs(block[1] - prev_block[1]) <= vertical_threshold or 
            abs(block[1] + block[3] - (prev_block[1] + prev_block[3])) <= vertical_threshold):
            # Check horizontal distance
            if (block[0] - (prev_block[0] + prev_block[2])) <= horizontal_threshold:
                current_group.append(block)
                continue
        
        # If we reach here, start a new group
        grouped_blocks.append(current_group)
        current_group = [block]
    
    # Add the last group
    grouped_blocks.append(current_group)
    
    # Merge blocks in each group into a single block
    merged_blocks = []
    for group in grouped_blocks:
        if not group:
            continue
            
        # Find the bounding box that contains all blocks in the group
        min_x = min(block[0] for block in group)
        min_y = min(block[1] for block in group)
        max_x = max(block[0] + block[2] for block in group)
        max_y = max(block[1] + block[3] for block in group)
        
        # Concatenate text with spaces
        merged_text = " ".join(block[4] for block in group)
        
        # Create the merged block
        merged_blocks.append((min_x, min_y, max_x - min_x, max_y - min_y, merged_text))
    
    return merged_blocks

def create_document_svg(image_path, header_boundary, header_elements, output_svg_path):
    """
    Create an SVG representation of the document with header and body regions
    
    Args:
        image_path: Path to the original document image
        header_boundary: Y-coordinate of the boundary between header and body
        header_elements: Dictionary of elements detected in the header
        output_svg_path: Path to save the output SVG
    """
    # Read the original image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Create SVG content with embedded image
    svg_content = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<style>
    .text-box {{ fill: rgba(0, 255, 0, 0.2); stroke: #00FF00; stroke-width: 1; }}
    .logo-box {{ fill: rgba(0, 0, 255, 0.2); stroke: #0000FF; stroke-width: 2; }}
    .table-box {{ fill: rgba(255, 0, 0, 0.2); stroke: #FF0000; stroke-width: 1; }}
    .header-region {{ fill: rgba(255, 255, 0, 0.05); stroke: #FFCC00; stroke-width: 2; stroke-dasharray: 5,5; }}
    .body-region {{ fill: rgba(100, 100, 255, 0.05); stroke: #AAAAFF; stroke-width: 2; stroke-dasharray: 5,5; }}
    .label {{ font-family: Arial; font-size: 12px; }}
    .header-label {{ fill: #AA6600; }}
    .text-label {{ fill: #008800; }}
    .logo-label {{ fill: #0000AA; }}
    .table-label {{ fill: #AA0000; }}
    .boundary-line {{ stroke: #FF0000; stroke-width: 2; }}
</style>

<!-- Background image reference -->
<image href="data:image/jpeg;base64,{image_to_base64(img)}" width="{width}" height="{height}" />

<!-- Header and Body regions -->
<rect x="0" y="0" width="{width}" height="{header_boundary}" class="header-region" />
<rect x="0" y="{header_boundary}" width="{width}" height="{height - header_boundary}" class="body-region" />
<line x1="0" y1="{header_boundary}" x2="{width}" y2="{header_boundary}" class="boundary-line" />
<text x="10" y="{header_boundary - 10}" class="label header-label">Header Region</text>
<text x="10" y="{header_boundary + 20}" class="label header-label">Body Region</text>

<!-- Header Elements -->
'''
    
    # Add header text elements
    for i, (x, y, w, h, text) in enumerate(header_elements["text"]):
        # Clean text for SVG (escape special characters)
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        truncated_text = safe_text[:30] + ("..." if len(safe_text) > 30 else "")
        
        svg_content += f'''<!-- Text block {i+1} -->
<rect x="{x}" y="{y}" width="{w}" height="{h}" class="text-box" />
<text x="{x + 2}" y="{y - 3}" class="label text-label">{truncated_text}</text>

'''
    
    # Add header logo elements
    for i, (x, y, w, h) in enumerate(header_elements["logos"]):
        svg_content += f'''<!-- Logo {i+1} -->
<rect x="{x}" y="{y}" width="{w}" height="{h}" class="logo-box" />
<text x="{x + 2}" y="{y - 3}" class="label logo-label">LOGO</text>

'''
    
    # Add header table elements
    for i, (x, y, w, h) in enumerate(header_elements["tables"]):
        svg_content += f'''<!-- Table {i+1} -->
<rect x="{x}" y="{y}" width="{w}" height="{h}" class="table-box" />
<text x="{x + 2}" y="{y - 3}" class="label table-label">TABLE</text>

'''
    
    # Close SVG
    svg_content += '</svg>'
    
    # Write to file
    with open(output_svg_path, 'w', encoding='utf-8') as svg_file:
        svg_file.write(svg_content)

def image_to_base64(image):
    """Convert an image to base64 string for embedding in SVG"""
    # Convert to JPEG format
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Reuse your detect_logo and detect_tables functions from the first code sample
def detect_logo(gray_image, height, width):
    """
    Detect logo in the document image (can be anywhere, not just top portion)
    """
    # Apply adaptive thresholding to isolate potential logo components
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    
    # Use edge detection to find regions with high edge density (likely logo)
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Dilate edges to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size, position, and density
    logo_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Logo typically has significant edge density and is square-ish
        if area > 2000 and h > 50 and w > 50:
            # Check edge density inside the region
            roi = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi > 0) / area
            
            # Check aspect ratio (logos are often more square-like)
            aspect_ratio = w / h if h > 0 else 0
            
            if edge_density > 0.05 and 0.5 < aspect_ratio < 2.0:
                logo_boxes.append((x, y, w, h))
    
    # Merge overlapping boxes
    logo_boxes = merge_overlapping_boxes(logo_boxes)
    
    # If multiple logos detected, keep the ones with highest edge density
    if len(logo_boxes) > 1:
        # Sort by edge density and size
        logo_densities = []
        for x, y, w, h in logo_boxes:
            roi = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi > 0) / (w * h)
            logo_densities.append((edge_density, w * h, (x, y, w, h)))
        
        logo_densities.sort(reverse=True)  # Sort by edge density (highest first)
        logo_boxes = [logo_densities[0][2]]  # Keep just the highest density one
    
    return logo_boxes

def detect_tables(gray_image, height, width):
    """
    Detect tables using line detection and structural analysis
    """
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine horizontal and vertical lines
    table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Dilate to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    table_mask = cv2.dilate(table_mask, kernel, iterations=3)
    
    # Apply closing to fill gaps
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours for potential tables
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and aspect ratio
    table_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Tables typically have reasonable aspect ratio and significant size
        if area > 10000 and 0.5 < aspect_ratio < 10:
            # Check if region has grid-like structure
            roi = table_mask[y:y+h, x:x+w]
            line_density = np.sum(roi > 0) / area
            
            if line_density > 0.05:  # Adjust threshold as needed
                table_boxes.append((x, y, w, h))
    
    # Merge overlapping boxes
    table_boxes = merge_overlapping_boxes(table_boxes)
    
    return table_boxes

def merge_overlapping_boxes(boxes):
    """
    Merge overlapping or close bounding boxes
    
    Args:
        boxes: List of tuples (x, y, w, h)
        
    Returns:
        List of merged boxes
    """
    if not boxes:
        return []
    
    # Sort boxes by y-coordinate (top to bottom)
    sorted_boxes = sorted(boxes, key=lambda box: box[1])
    merged_boxes = [sorted_boxes[0]]
    
    for current_box in sorted_boxes[1:]:
        prev_box = merged_boxes[-1]
        
        # Extract coordinates
        prev_x, prev_y, prev_w, prev_h = prev_box
        curr_x, curr_y, curr_w, curr_h = current_box
        
        # Calculate boundaries
        prev_right = prev_x + prev_w
        prev_bottom = prev_y + prev_h
        curr_right = curr_x + curr_w
        curr_bottom = curr_y + curr_h
        
        # Check for overlap or close proximity (within 20 pixels)
        if (prev_x - 20 <= curr_right and curr_x - 20 <= prev_right and
            prev_y - 20 <= curr_bottom and curr_y - 20 <= prev_bottom):
            # Merge the boxes
            new_x = min(prev_x, curr_x)
            new_y = min(prev_y, curr_y)
            new_right = max(prev_right, curr_right)
            new_bottom = max(prev_bottom, curr_bottom)
            
            merged_boxes[-1] = (new_x, new_y, new_right - new_x, new_bottom - new_y)
        else:
            merged_boxes.append(current_box)
    
    return merged_boxes

# Function to identify specific elements in the header
def identify_header_elements(header_elements):
    """
    Identify specific elements in the header such as institution name, course code, etc.
    
    Args:
        header_elements: Dictionary of elements detected in the header
    
    Returns:
        Dictionary of identified header components with their positions
    """
    header_components = {
        "institution": None,
        "course_code": None,
        "exam_title": None,
        "date": None
    }
    
    # Sort text blocks by y-position (top to bottom)
    text_blocks = sorted(header_elements["text"], key=lambda x: x[1])
    
    # Simple heuristics for identification:
    # 1. Institution is typically at the top
    # 2. Course code often contains digits and capital letters
    # 3. Exam title might contain keywords like "Exam", "Test", "Quiz"
    # 4. Date may contain date formats
    
    # Define regex patterns for identification
    date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', re.IGNORECASE)
    course_code_pattern = re.compile(r'\b[A-Z]{2,4}\s*\d{3,4}\b')
    exam_keywords = ["exam", "test", "quiz", "assessment", "final", "midterm"]
    
    for x, y, w, h, text in text_blocks:
        # Check for institution (typically first or second text block)
        if header_components["institution"] is None and y < header_elements["text"][0][1] + 100:
            header_components["institution"] = (x, y, w, h, text)
            continue
            
        # Check for course code
        if header_components["course_code"] is None and course_code_pattern.search(text):
            header_components["course_code"] = (x, y, w, h, text)
            continue
            
        # Check for exam title
        if header_components["exam_title"] is None and any(keyword in text.lower() for keyword in exam_keywords):
            header_components["exam_title"] = (x, y, w, h, text)
            continue
            
        # Check for date
        if header_components["date"] is None and date_pattern.search(text):
            header_components["date"] = (x, y, w, h, text)
            continue
    
    return header_components

# Main function to integrate everything
def generate_question_paper_svg(image_path, output_svg_path):
    """
    Generate an SVG representation of a question paper with header and body clearly marked
    
    Args:
        image_path: Path to the question paper image
        output_svg_path: Path to save the output SVG
    """
    process_question_paper_to_svg(image_path, output_svg_path)
    print(f"SVG generated successfully at {output_svg_path}")