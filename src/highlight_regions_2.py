import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
import re
from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, fallback_header_detection

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




#NEW ADDITIONS ------------------------
def convert_to_ideal_format(header_elements, original_image, header_y_offset=0):
    """
    Convert current detection format to the ideal SVG-ready format
    
    Args:
        header_elements: Current format dictionary with text/logos/tables
        original_image: Original image for font size estimation
        header_y_offset: Y offset if this is header region (0 for full image)
    
    Returns:
        List of dictionaries in ideal format for SVG generation
    """
    processed_elements = []
    element_id_counter = 1
    
    # Convert text elements
    for i, (x, y, w, h, text) in enumerate(header_elements.get("text", [])):
        # Adjust coordinates if this is a header region
        actual_y = y + header_y_offset
        
        # Estimate font properties (basic heuristics)
        font_size = estimate_font_size(h)
        font_weight, font_style = estimate_font_style(text, original_image, x, actual_y, w, h)
        baseline_y = estimate_baseline(actual_y, h, font_size)
        
        text_element = {
            "type": "text_block",
            "id": f"text_block_{element_id_counter}",
            "bbox": {
                "x": int(x),
                "y": int(actual_y),
                "width": int(w),
                "height": int(h)
            },
            "text_content": clean_text_content(text),
            "font_family": "Arial",  # Default, could be improved with font detection
            "font_size": f"{font_size}px",
            "font_weight": font_weight,
            "font_style": font_style,
            "text_color": "black",  # Default, could add color detection
            "baseline_y": baseline_y
        }
        processed_elements.append(text_element)
        element_id_counter += 1
    
    # Convert logo elements to image regions
    for i, (x, y, w, h) in enumerate(header_elements.get("logos", [])):
        actual_y = y + header_y_offset
        
        logo_element = {
            "type": "image_region",
            "id": f"logo_{element_id_counter}",
            "bbox": {
                "x": int(x),
                "y": int(actual_y),
                "width": int(w),
                "height": int(h)
            }
        }
        processed_elements.append(logo_element)
        element_id_counter += 1
    
    # Convert table elements (simplified - would need line detection for full tables)
    for i, (x, y, w, h) in enumerate(header_elements.get("tables", [])):
        actual_y = y + header_y_offset
        
        # For now, represent table as a rectangular border
        table_border_elements = create_table_border_lines(x, actual_y, w, h, element_id_counter)
        processed_elements.extend(table_border_elements)
        element_id_counter += len(table_border_elements)
    
    return processed_elements

def estimate_font_size(text_height):
    """Estimate font size from text height"""
    # Rule of thumb: font size is roughly 70-80% of text height
    return max(8, int(text_height * 0.75))

def estimate_font_style(text, image, x, y, w, h):
    """
    Estimate if text is bold or italic using basic image analysis
    
    Args:
        text: The text content
        image: Original image
        x, y, w, h: Bounding box coordinates
    
    Returns:
        Tuple of (font_weight, font_style)
    """
    try:
        # Extract text region
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        text_region = gray[y:y+h, x:x+w]
        
        # Simple bold detection: measure stroke thickness
        # Binarize the text region
        _, binary = cv2.threshold(text_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert so text is white on black
        binary = cv2.bitwise_not(binary)
        
        # Calculate average stroke width
        if np.sum(binary) > 0:  # If there's any text
            # Use morphological operations to estimate stroke width
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            eroded = cv2.erode(binary, kernel, iterations=1)
            stroke_ratio = np.sum(eroded) / np.sum(binary) if np.sum(binary) > 0 else 0
            
            # If a lot of pixels remain after erosion, text is likely bold
            font_weight = "bold" if stroke_ratio > 0.3 else "normal"
        else:
            font_weight = "normal"
        
        # Simple italic detection: check for slant (very basic)
        # This is quite complex and unreliable, so defaulting to normal
        font_style = "normal"
        
        # Alternative: check for ALL CAPS as a hint for bold
        if text.isupper() and len(text) > 2:
            font_weight = "bold"
            
    except Exception as e:
        # Default values if analysis fails
        font_weight = "normal"
        font_style = "normal"
    
    return font_weight, font_style

def estimate_baseline(y, height, font_size):
    """Estimate baseline position for text"""
    # Baseline is typically around 80% down from the top of the text box
    return int(y + height * 0.8)

def clean_text_content(text):
    """Clean and normalize text content"""
    # Remove extra whitespace and normalize
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return cleaned

def create_table_border_lines(x, y, width, height, start_id):
    """
    Create line elements for table borders
    
    Args:
        x, y, width, height: Table bounding box
        start_id: Starting ID for line elements
    
    Returns:
        List of line elements forming table border
    """
    lines = []
    
    # Top border
    lines.append({
        "type": "line_horizontal",
        "id": f"table_line_{start_id}",
        "start": {"x": x, "y": y},
        "end": {"x": x + width, "y": y},
        "stroke_width": 1,
        "stroke_color": "black"
    })
    
    # Bottom border
    lines.append({
        "type": "line_horizontal", 
        "id": f"table_line_{start_id + 1}",
        "start": {"x": x, "y": y + height},
        "end": {"x": x + width, "y": y + height},
        "stroke_width": 1,
        "stroke_color": "black"
    })
    
    # Left border
    lines.append({
        "type": "line_vertical",
        "id": f"table_line_{start_id + 2}", 
        "start": {"x": x, "y": y},
        "end": {"x": x, "y": y + height},
        "stroke_width": 1,
        "stroke_color": "black"
    })
    
    # Right border
    lines.append({
        "type": "line_vertical",
        "id": f"table_line_{start_id + 3}",
        "start": {"x": x + width, "y": y},
        "end": {"x": x + width, "y": y + height},
        "stroke_width": 1,
        "stroke_color": "black"
    })
    
    return lines

def add_line_detection(image, processed_elements):
    """
    Add detected lines to the processed elements
    
    Args:
        image: Input image
        processed_elements: Existing list of elements
    
    Returns:
        Updated list with line elements added
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical lines  
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    line_id = len(processed_elements) + 1
    
    # Process horizontal lines
    h_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in h_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 30:  # Filter short lines
            line_element = {
                "type": "line_horizontal",
                "id": f"hline_{line_id}",
                "start": {"x": x, "y": y + h//2},
                "end": {"x": x + w, "y": y + h//2},
                "stroke_width": max(1, h),
                "stroke_color": "black"
            }
            processed_elements.append(line_element)
            line_id += 1
    
    # Process vertical lines
    v_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in v_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 30:  # Filter short lines
            line_element = {
                "type": "line_vertical",
                "id": f"vline_{line_id}",
                "start": {"x": x + w//2, "y": y},
                "end": {"x": x + w//2, "y": y + h},
                "stroke_width": max(1, w),
                "stroke_color": "black"
            }
            processed_elements.append(line_element)
            line_id += 1
    
    return processed_elements

def print_ideal_format(processed_elements):
    """Print the processed elements in a readable format"""
    print("=" * 60)
    print("PROCESSED ELEMENTS (SVG-Ready Format)")
    print("=" * 60)
    
    for i, element in enumerate(processed_elements):
        print(f"\nElement {i+1}:")
        print(f"  Type: {element['type']}")
        print(f"  ID: {element['id']}")
        
        if 'bbox' in element:
            bbox = element['bbox']
            print(f"  Bounding Box: ({bbox['x']}, {bbox['y']}) - {bbox['width']}x{bbox['height']}")
        
        if element['type'] == 'text_block':
            print(f"  Text: '{element['text_content']}'")
            print(f"  Font: {element['font_family']} {element['font_size']} {element['font_weight']} {element['font_style']}")
            print(f"  Baseline Y: {element['baseline_y']}")
        
        elif element['type'] in ['line_horizontal', 'line_vertical']:
            start = element['start']
            end = element['end']
            print(f"  Line: ({start['x']}, {start['y']}) -> ({end['x']}, {end['y']})") 
            print(f"  Stroke: {element['stroke_width']}px {element['stroke_color']}")
        
        elif element['type'] == 'image_region':
            print(f"  Image region (logo/figure)")

#NEW HIGHLIGHTING FUNCTIONS -------------------------------------------

def create_highlighted_image_svg_format(image_path, header_boundary, processed_elements, output_image_path):
    """
    Create a highlighted image using SVG-ready format (processed_elements)
    Shows much more detail than the original highlighting function
    
    Args:
        image_path: Path to the original document image
        header_boundary: Y-coordinate of the boundary between header and body
        processed_elements: List of SVG-ready elements with type, bbox, etc.
        output_image_path: Path to save the highlighted image
    """
    # Read the original image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Create a copy for highlighting
    highlighted_img = img.copy()
    
    # Define colors for different elements (BGR format)
    colors = {
        'header_region': (0, 255, 255),      # Yellow
        'body_region': (255, 100, 100),      # Light blue
        'text_block': (0, 255, 0),           # Green
        'image_region': (255, 0, 0),         # Blue (for logos)
        'line_horizontal': (0, 165, 255),    # Orange
        'line_vertical': (0, 0, 0, 0),      # Magenta
        'table_line': (0, 0, 255),           # Red (for table borders)
        'boundary': (0, 0, 255),             # Red
        'text_label': (0, 150, 0),           # Dark green
        'line_label': (0, 100, 200),         # Dark orange
        'logo_label': (150, 0, 0),           # Dark blue
    }
    
    # Create overlay for semi-transparent regions
    overlay = img.copy()
    
    # Highlight header and body regions
    if header_boundary > 0:
        cv2.rectangle(overlay, (0, 0), (width, header_boundary), colors['header_region'], -1)
        cv2.rectangle(overlay, (0, header_boundary), (width, height), colors['body_region'], -1)
        
        # Blend overlay with original image for transparency effect
        alpha = 0.08  # Light transparency
        highlighted_img = cv2.addWeighted(highlighted_img, 1 - alpha, overlay, alpha, 0)
        
        # Draw boundary line
        cv2.line(highlighted_img, (0, header_boundary), (width, header_boundary), colors['boundary'], 3)
        
        # Add region labels
        cv2.putText(highlighted_img, "HEADER REGION", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors['boundary'], 2)
        cv2.putText(highlighted_img, "BODY REGION", (10, header_boundary + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors['boundary'], 2)
    
    # Process each element in the SVG format
    line_counter = {'horizontal': 0, 'vertical': 0, 'table': 0}
    text_counter = 0
    logo_counter = 0
    
    for element in processed_elements:
        element_type = element.get('type', 'unknown')
        element_id = element.get('id', 'no_id')
        
        if element_type == 'text_block':
            text_counter += 1
            bbox = element['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            text_content = element.get('text_content', '')
            font_size = element.get('font_size', '12px')
            font_weight = element.get('font_weight', 'normal')
            baseline_y = element.get('baseline_y', y + h)
            
            # Draw text bounding box
            cv2.rectangle(highlighted_img, (x, y), (x + w, y + h), colors['text_block'], 2)
            
            # Draw baseline indicator
            cv2.line(highlighted_img, (x, baseline_y), (x + w, baseline_y), colors['text_block'], 1)
            
            # Add text label with content and font info
            safe_text = text_content.replace('\n', ' ').replace('\r', ' ')
            truncated_text = safe_text[:15] + ("..." if len(safe_text) > 15 else "")
            weight_indicator = "B" if font_weight == "bold" else ""
            label = f"T{text_counter}: {truncated_text} ({font_size}{weight_indicator})"
            
            # Position label above the box
            label_y = max(y - 8, 15)
            cv2.putText(highlighted_img, label, (x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['text_label'], 1)
            
            # Add small circle at baseline start
            cv2.circle(highlighted_img, (x, baseline_y), 3, colors['text_block'], -1)
        
        elif element_type == 'image_region':
            logo_counter += 1
            bbox = element['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw logo bounding box with thicker border
            cv2.rectangle(highlighted_img, (x, y), (x + w, y + h), colors['image_region'], 3)
            
            # Add diagonal lines to indicate image region
            cv2.line(highlighted_img, (x, y), (x + w, y + h), colors['image_region'], 2)
            cv2.line(highlighted_img, (x + w, y), (x, y + h), colors['image_region'], 2)
            
            # Add label
            label = f"LOGO {logo_counter} ({w}x{h})"
            label_y = max(y - 8, 15)
            cv2.putText(highlighted_img, label, (x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['logo_label'], 2)
        
        elif element_type == 'line_horizontal':
            continue
            start = element['start']
            end = element['end']
            stroke_width = element.get('stroke_width', 1)
            
            # Determine if it's a table line or regular line
            if 'table_line' in element_id:
                line_counter['table'] += 1
                color = colors['table_line']
                label_prefix = f"TBL{line_counter['table']}"
            else:
                line_counter['horizontal'] += 1
                color = colors['line_horizontal']
                label_prefix = f"H{line_counter['horizontal']}"
            
            # Draw the line with appropriate thickness
            cv2.line(highlighted_img, (start['x'], start['y']), (end['x'], end['y']), 
                     color, max(2, stroke_width))
            
            # Add small circles at line endpoints
            cv2.circle(highlighted_img, (start['x'], start['y']), 4, color, -1)
            cv2.circle(highlighted_img, (end['x'], end['y']), 4, color, -1)
            
            # Add label at line midpoint
            mid_x = (start['x'] + end['x']) // 2
            mid_y = start['y']
            label = f"{label_prefix} ({end['x'] - start['x']}px)"
            cv2.putText(highlighted_img, label, (mid_x - 30, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['line_label'], 1)
        
        elif element_type == 'line_vertical':
            continue
            start = element['start']
            end = element['end']
            stroke_width = element.get('stroke_width', 1)
            
            # Determine if it's a table line or regular line
            if 'table_line' in element_id:
                line_counter['table'] += 1
                color = colors['table_line']
                label_prefix = f"TBL{line_counter['table']}"
            else:
                line_counter['vertical'] += 1
                color = colors['line_vertical']
                label_prefix = f"V{line_counter['vertical']}"
            
            # Draw the line
            cv2.line(highlighted_img, (start['x'], start['y']), (end['x'], end['y']), 
                     color, max(2, stroke_width))
            
            # Add small circles at line endpoints
            cv2.circle(highlighted_img, (start['x'], start['y']), 4, color, -1)
            cv2.circle(highlighted_img, (end['x'], end['y']), 4, color, -1)
            
            # Add label at line midpoint
            mid_x = start['x']
            mid_y = (start['y'] + end['y']) // 2
            label = f"{label_prefix} ({end['y'] - start['y']}px)"
            
            # Rotate label for vertical lines or place it to the side
            cv2.putText(highlighted_img, label, (mid_x + 10, mid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['line_label'], 1)
    
    # Add legend
    add_legend(highlighted_img, colors, line_counter, text_counter, logo_counter)
    
    # Save the highlighted image
    cv2.imwrite(output_image_path, highlighted_img)

def add_legend(image, colors, line_counter, text_counter, logo_counter):
    """
    Add a legend showing what each color represents and counts
    """
    legend_start_x = 10
    legend_start_y = image.shape[0] - 200  # Start from bottom
    
    legend_items = [
        (f"Text Blocks: {text_counter}", colors['text_block']),
        (f"Logos: {logo_counter}", colors['image_region']),
        (f"Horizontal Lines: {line_counter['horizontal']}", colors['line_horizontal']),
        (f"Vertical Lines: {line_counter['vertical']}", colors['line_vertical']),
        (f"Table Lines: {line_counter['table']}", colors['table_line']),
    ]
    
    # Draw legend background
    legend_height = len(legend_items) * 25 + 20
    legend_width = 250
    
    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, (legend_start_x - 5, legend_start_y - 10), 
                  (legend_start_x + legend_width, legend_start_y + legend_height), 
                  (255, 255, 255), -1)
    cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
    
    # Draw legend border
    cv2.rectangle(image, (legend_start_x - 5, legend_start_y - 10), 
                  (legend_start_x + legend_width, legend_start_y + legend_height), 
                  (0, 0, 0), 2)
    
    # Add legend title
    cv2.putText(image, "LEGEND", (legend_start_x, legend_start_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add legend items
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_start_y + 25 + (i * 25)
        
        # Draw color indicator
        cv2.rectangle(image, (legend_start_x, y_pos - 10), 
                      (legend_start_x + 20, y_pos + 5), color, -1)
        cv2.rectangle(image, (legend_start_x, y_pos - 10), 
                      (legend_start_x + 20, y_pos + 5), (0, 0, 0), 1)
        
        # Add text
        cv2.putText(image, text, (legend_start_x + 30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



# Updated main function
def analyze_and_convert_to_svg_format(image_path, output_image_path, header_boundary=None):
    """
    Analyze image and return SVG-ready format
    
    Args:
        image_path: Path to image
        header_boundary: Y coordinate of header boundary (if splitting header/body)
    
    Returns:
        List of elements in SVG-ready format
    """
    # Load original image
    original_img = cv2.imread(image_path)
    header_img, body_img, header_boundary = detect_header_with_instructions_and_show_boxes(
        image_path, 
        header_path='temp_header.jpg', 
        body_path='temp_body.jpg', 
        split_image_path='temp_split'
    )

    header_elements = analyze_header_layout(header_img)
    processed_elements = convert_to_ideal_format(header_elements, header_img, 0)
    processed_elements = add_line_detection(header_img, processed_elements)
    
    create_highlighted_image_svg_format(
        image_path, 
        header_boundary, 
        processed_elements, 
        output_image_path
    )
   
     # Clean up temporary files
    for temp_file in ['temp_header.jpg', 'temp_body.jpg', 'temp_split/temp_boundary.jpg']:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"SVG-format highlighted image saved at {output_image_path}")
    print(f"Found {len(processed_elements)} total elements")
    return processed_elements


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

