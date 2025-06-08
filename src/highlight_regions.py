import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
import re
from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, fallback_header_detection
from create_svg import create_structured_svg_with_groups, create_svg_from_header_elements

def process_question_paper_with_highlights(image_path, output_image_path):
    """
    Process a question paper image to detect and highlight different regions
    
    Args:
        image_path: Path to the question paper image
        output_image_path: Path to save the highlighted output image
    """
    # Step 1: Split the image into header and body using your existing code
    header_img, body_img, header_boundary = detect_header_with_instructions_and_show_boxes(
        image_path, 
        header_path='temp_header.jpg', 
        body_path='temp_body.jpg', 
        split_image_path='temp_split'
    )
    
    # Step 2: Process the header using document layout analysis and optionally print the returned dictionary
    header_elements = analyze_header_layout(header_img)
    print_detected_elements(header_elements)

    # Step 3: Create highlighted image with detected regions
    create_highlighted_image(
        image_path, 
        header_boundary, 
        header_elements, 
        output_image_path
    )
    
    create_structured_svg_with_groups(image_path, header_boundary, header_elements, output_svg_path = 'output.svg')
    create_svg_from_header_elements(image_path, header_boundary, header_elements, output_svg_path = 'output2.svg')
    # Clean up temporary files
    for temp_file in ['temp_header.jpg', 'temp_body.jpg', 'temp_split/temp_boundary.jpg']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"Highlighted image saved at {output_image_path}")

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
    
    # Detect logo using edge detection
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

def create_highlighted_image(image_path, header_boundary, header_elements, output_image_path):
    """
    Create a highlighted image showing header and body regions with detected elements
    
    Args:
        image_path: Path to the original document image
        header_boundary: Y-coordinate of the boundary between header and body
        header_elements: Dictionary of elements detected in the header
        output_image_path: Path to save the highlighted image
    """
    # Read the original image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Create a copy for highlighting
    highlighted_img = img.copy()
    
    # Define colors for different elements (BGR format)
    colors = {
        'header_region': (0, 255, 255, 30),    # Yellow with transparency
        'body_region': (255, 100, 100, 30),    # Light blue with transparency
        'text': (0, 255, 0),                   # Green
        'logo': (255, 0, 0),                   # Blue
        'table': (0, 0, 255),                  # Red
        'boundary': (0, 0, 255)                # Red
    }
    
    # Create overlay for semi-transparent regions
    overlay = img.copy()
    
    # Highlight header region
    cv2.rectangle(overlay, (0, 0), (width, header_boundary), colors['header_region'][:3], -1)
    
    # Highlight body region
    cv2.rectangle(overlay, (0, header_boundary), (width, height), colors['body_region'][:3], -1)
    
    # Blend overlay with original image for transparency effect
    alpha = 0.1  # Transparency factor
    highlighted_img = cv2.addWeighted(highlighted_img, 1 - alpha, overlay, alpha, 0)
    
    # Draw boundary line between header and body
    cv2.line(highlighted_img, (0, header_boundary), (width, header_boundary), colors['boundary'], 3)
    
    # Add region labels
    cv2.putText(highlighted_img, "HEADER REGION", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, colors['boundary'], 2)
    cv2.putText(highlighted_img, "BODY REGION", (10, header_boundary + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, colors['boundary'], 2)
    
    # Highlight text elements in header
    for i, (x, y, w, h, text) in enumerate(header_elements["text"]):
        # Draw bounding box
        cv2.rectangle(highlighted_img, (x, y), (x + w, y + h), colors['text'], 2)
        
        # Add label with truncated text
        safe_text = text.replace('\n', ' ').replace('\r', ' ')
        truncated_text = safe_text[:20] + ("..." if len(safe_text) > 20 else "")
        label = f"TEXT: {truncated_text}"
        
        # Position label above the box
        label_y = max(y - 5, 15)
        cv2.putText(highlighted_img, label, (x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
    
    # Highlight logo elements in header
    for i, (x, y, w, h) in enumerate(header_elements["logos"]):
        # Draw bounding box
        cv2.rectangle(highlighted_img, (x, y), (x + w, y + h), colors['logo'], 3)
        
        # Add label
        label = f"LOGO {i+1}"
        label_y = max(y - 5, 15)
        cv2.putText(highlighted_img, label, (x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['logo'], 2)
    
    # Highlight table elements in header
    for i, (x, y, w, h) in enumerate(header_elements["tables"]):
        # Draw bounding box
        cv2.rectangle(highlighted_img, (x, y), (x + w, y + h), colors['table'], 2)
        
        # Add label
        label = f"TABLE {i+1}"
        label_y = max(y - 5, 15)
        cv2.putText(highlighted_img, label, (x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['table'], 2)
    
    # Save the highlighted image
    cv2.imwrite(output_image_path, highlighted_img)

def detect_logo(gray_image, height, width):
    """
    Detect logo in the document image using edge density
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
        logo_densities = []
        for x, y, w, h in logo_boxes:
            roi = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi > 0) / (w * h)
            logo_densities.append((edge_density, w * h, (x, y, w, h)))
        
        logo_densities.sort(reverse=True)
        logo_boxes = [logo_densities[0][2]]
    
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
            
            if line_density > 0.05:
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
    
    # Define regex patterns for identification
    date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', re.IGNORECASE)
    course_code_pattern = re.compile(r'\b[A-Z]{2,4}\s*\d{3,4}\b')
    exam_keywords = ["exam", "test", "quiz", "assessment", "final", "midterm"]
    
    for x, y, w, h, text in text_blocks:
        # Check for institution (typically first or second text block)
        if header_components["institution"] is None and y < text_blocks[0][1] + 100:
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


#Function to print out the dictionary of detected elements 
def print_detected_elements(header_elements):
    """Print detected elements in a readable format"""
    
    print("=" * 50)
    print("DETECTED ELEMENTS")
    print("=" * 50)
    
    # Print text elements
    print(f"\nTEXT BLOCKS ({len(header_elements['text'])} found):")
    print("-" * 30)
    for i, (x, y, w, h, text) in enumerate(header_elements['text']):
        print(f"Text {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w} x {h}")
        print(f"  Content: '{text}'")
        print()
    
    # Print logo elements
    print(f"LOGOS ({len(header_elements['logos'])} found):")
    print("-" * 30)
    for i, (x, y, w, h) in enumerate(header_elements['logos']):
        print(f"Logo {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w} x {h}")
        print()
    
    # Print table elements
    print(f"TABLES ({len(header_elements['tables'])} found):")
    print("-" * 30)
    for i, (x, y, w, h) in enumerate(header_elements['tables']):
        print(f"Table {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w} x {h}")
        print()



# Main function to process the image with highlights
def generate_highlighted_question_paper(image_path, output_image_path):
    """
    Generate a highlighted image of a question paper with regions and elements marked
    
    Args:
        image_path: Path to the question paper image
        output_image_path: Path to save the highlighted image
    """
    process_question_paper_with_highlights(image_path, output_image_path)

    print(f"Highlighted image generated successfully at {output_image_path}")

# Example usage:
# generate_highlighted_question_paper("question_paper.jpg", "highlighted_question_paper.jpg")

