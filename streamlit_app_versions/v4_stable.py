import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import shutil
import pytesseract
from pytesseract import Output
import re

# Page configuration
st.set_page_config(
    page_title="Algorithmic Layout Analysis Demo",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Algorithmic Layout Analysis Pipeline")
st.markdown("Upload a document image to see algorithmic layout detection in action!")

# Create temporary directories
TEMP_DIR = Path("temp_algorithmic")
INPUT_DIR = TEMP_DIR / "input"
OUTPUT_DIR = TEMP_DIR / "output"
HEADERS_DIR = OUTPUT_DIR / "headers"
BODIES_DIR = OUTPUT_DIR / "bodies"
SPLIT_DIR = OUTPUT_DIR / "split_images"

# Setup directories
for dir_path in [TEMP_DIR, INPUT_DIR, OUTPUT_DIR, HEADERS_DIR, BODIES_DIR, SPLIT_DIR]:
    dir_path.mkdir(exist_ok=True)

def detect_header_with_instructions_and_show_boxes(image_path, header_path='headers/test.jpg', body_path='bodies/test.jpg', split_image_path='split_images'):
    """
    Your header/body detection function - imported from your code
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to improve OCR results
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use pytesseract to extract text with bounding boxes
    custom_config = r'--oem 3 --psm 1'  # Page segmentation mode: 1 = Automatic page segmentation with OSD
    data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)
    
    # Find the "instructions to candidates" line
    instruction_line_y = None
    instruction_regex = re.compile(r'instructions\s+to\s+candidates', re.IGNORECASE)
    instruction_regex_alt = re.compile(r'candidates', re.IGNORECASE)
    
    # Group text by line (using top coordinate and height)
    lines = {}
    detected_texts = []  # Store all detected text for display
    
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:  # Only consider text with decent confidence
            text = data['text'][i]
            if text.strip():  # Skip empty text
                top = data['top'][i]
                height = data['height'][i]
                left = data['left'][i]
                width = data['width'][i]
                conf = data['conf'][i]
                
                # Store for display
                detected_texts.append({
                    'text': text,
                    'bbox': (left, top, width, height),
                    'confidence': conf
                })
                
                # Group by line (allow for small variations in top position)
                line_id = top // 10  # Group lines within 10 pixels
                if line_id not in lines:
                    lines[line_id] = {
                        'texts': [],
                        'top': top,
                        'bottom': top + height,
                        'bboxes': []
                    }
                lines[line_id]['texts'].append(text)
                lines[line_id]['bottom'] = max(lines[line_id]['bottom'], top + height)
                lines[line_id]['bboxes'].append((left, top, width, height))
    
    # Search for instruction line in each detected line of text
    instruction_found = False
    for line_id, line_data in lines.items():
        line_text = ' '.join(line_data['texts']).lower()
        if (instruction_regex.search(line_text) or instruction_regex_alt.search(line_text)):
            instruction_line_y = line_data['bottom']
            instruction_found = True
            break
    
    # If we found the instruction line, use it as boundary
    if instruction_line_y:
        header_boundary = instruction_line_y
    else:
        # Simple fallback - use 1/3 of image height
        header_boundary = h // 3
    
    # Create header and body images
    header_img = img[0:header_boundary, :]
    body_img = img[header_boundary:, :]
    
    # Save header and body
    cv2.imwrite(header_path, header_img)
    cv2.imwrite(body_path, body_img)
    
    # Drawing boundary on copy image 
    img_with_line = img.copy()
    cv2.line(img_with_line, (0, header_boundary), (img.shape[1], header_boundary), (0, 0, 255), 3)
    cv2.putText(img_with_line, "Header/Body Boundary", (10, header_boundary - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Save boundary visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    boundary_path = os.path.join(split_image_path, f"{base_name}_boundary.jpg")
    cv2.imwrite(boundary_path, img_with_line)
    
    return {
        'header_img': header_img,
        'body_img': body_img,
        'boundary_img': img_with_line,
        'header_boundary': header_boundary,
        'instruction_found': instruction_found,
        'detected_texts': detected_texts,
        'lines': lines,
        'paths': {
            'header': header_path,
            'body': body_path,
            'boundary': boundary_path
        }
    }

def process_body(image_path, output_path, adaptive_block_size=11, adaptive_c=5, min_width=80, min_height=20):
    """
    Process body region to detect text elements using contour detection
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
    
    # Optional morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store bounding boxes
    boxes = []
    
    # Loop through contours to get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_width and h > min_height:  # Filter based on size
            boxes.append((x, y, w, h))
    
    # Create visualization
    img_with_boxes = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the output image with bounding boxes
    cv2.imwrite(output_path, img_with_boxes)
    
    return {
        'original': img,
        'gray': gray,
        'thresh': thresh,
        'dilated': dilated,
        'boxes': boxes,
        'visualization': img_with_boxes,
        'output_path': output_path
    }

def analyze_document_layout(image_path, output_path, display_steps=False):
    """
    Enhanced document layout analysis for header region that identifies and draws 
    bounding boxes around text, logos, and tables.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None, {}
    
    # Create a copy for drawing results
    result_img = original_img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Create a mask for tracking detected regions (to avoid overlaps)
    mask_detected = np.zeros((height, width), dtype=np.uint8)
    
    # Dictionary to store detected elements
    detected_elements = {
        "text": [],
        "logos": [],
        "tables": []
    }
    
    # Step 1: Detect and extract the logo
    logo_boxes = detect_logo(gray, height, width)
    
    # Add logo regions to the detected mask
    for x, y, w, h in logo_boxes:
        cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
        detected_elements["logos"].append((x, y, w, h))
        # Draw red bounding box around logo
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result_img, "Logo", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Step 2: Detect tables using line detection techniques
    table_boxes = detect_tables(gray, height, width)
    
    # Filter out tables that overlap with already detected elements
    for box in table_boxes:
        x, y, w, h = box
        table_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(table_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(table_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h) if (w * h) > 0 else 0
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["tables"].append(box)
            # Draw blue bounding box around table
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_img, "Table", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Step 3: Detect text with improved parameters
    text_boxes = detect_text_improved_header(gray, height, width, mask_detected)
    
    # Filter out text regions that overlap with already detected elements
    for box in text_boxes:
        x, y, w, h = box
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(text_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(text_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h) if (w * h) > 0 else 0
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["text"].append(box)
            # Draw green bounding box around text
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_img, "Text", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite(output_path, result_img)
    
    return {
        'result_img': result_img,
        'detected_elements': detected_elements,
        'gray': gray,
        'mask_detected': mask_detected
    }
    """
    Enhanced document layout analysis for header region that identifies and draws 
    bounding boxes around text, logos, and tables.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None, {}
    
    # Create a copy for drawing results
    result_img = original_img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Create a mask for tracking detected regions (to avoid overlaps)
    mask_detected = np.zeros((height, width), dtype=np.uint8)
    
    # Dictionary to store detected elements
    detected_elements = {
        "text": [],
        "logos": [],
        "tables": []
    }
    
    # Step 1: Detect and extract the logo
    logo_boxes = detect_logo(gray, height, width)
    
    # Add logo regions to the detected mask
    for x, y, w, h in logo_boxes:
        cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
        detected_elements["logos"].append((x, y, w, h))
        # Draw red bounding box around logo
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result_img, "Logo", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Step 2: Detect tables using line detection techniques
    table_boxes = detect_tables(gray, height, width)
    
    # Filter out tables that overlap with already detected elements
    for box in table_boxes:
        x, y, w, h = box
        table_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(table_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(table_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h) if (w * h) > 0 else 0
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["tables"].append(box)
            # Draw blue bounding box around table
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_img, "Table", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Step 3: Detect text with improved parameters
    text_boxes = detect_text_improved_header(gray, height, width, mask_detected)
    
    # Filter out text regions that overlap with already detected elements
    for box in text_boxes:
        x, y, w, h = box
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(text_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(text_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h) if (w * h) > 0 else 0
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["text"].append(box)
            # Draw green bounding box around text
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_img, "Text", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite(output_path, result_img)
    
    return {
        'result_img': result_img,
        'detected_elements': detected_elements,
        'gray': gray,
        'mask_detected': mask_detected
    }

def detect_logo(gray_image, height, width):
    """Detect logo using edge density and aspect ratio"""
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    
    # Use edge detection
    edges = cv2.Canny(gray_image, 50, 150)
    
    # Dilate edges to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for logo characteristics
    logo_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Logo criteria: significant size, edge density, square-ish aspect ratio
        if area > 2000 and h > 50 and w > 50:
            # Check edge density
            roi = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi > 0) / area if area > 0 else 0
            
            # Check aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            if edge_density > 0.05 and 0.5 < aspect_ratio < 2.0:
                logo_boxes.append((x, y, w, h))
    
    return merge_overlapping_boxes(logo_boxes)

def detect_tables(gray_image, height, width):
    """Detect tables using line detection"""
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
    
    # Dilate and close to connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    table_mask = cv2.dilate(table_mask, kernel, iterations=3)
    table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours for potential tables
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for table characteristics
    table_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Table criteria: significant size and reasonable aspect ratio
        if area > 10000 and 0.5 < aspect_ratio < 10:
            # Check line density
            roi = table_mask[y:y+h, x:x+w]
            line_density = np.sum(roi > 0) / area if area > 0 else 0
            
            if line_density > 0.05:
                table_boxes.append((x, y, w, h))
    
    return merge_overlapping_boxes(table_boxes)

def detect_text_improved_header(gray_image, height, width, existing_mask=None):
    """Improved text detection specifically tuned for header regions"""
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Exclude existing mask regions
    if existing_mask is not None:
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(existing_mask))
    
    # Create MSER detector for text regions
    try:
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)
        
        # Create mask for MSER regions
        mser_mask = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mser_mask, [hull], 0, 255, -1)
        
        # Combine with binary image
        text_mask = cv2.bitwise_or(binary, mser_mask)
    except:
        # Fallback if MSER fails
        text_mask = binary
    
    # Connect characters and words
    kernel_connect_chars = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    word_mask = cv2.dilate(text_mask, kernel_connect_chars, iterations=1)
    
    kernel_connect_words = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    line_mask = cv2.dilate(word_mask, kernel_connect_words, iterations=1)
    
    # Find contours for text lines
    contours, _ = cv2.findContours(line_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for text characteristics
    line_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Text line criteria
        if area > 500 and w > 20 and h > 8:
            # Check text density
            roi = text_mask[y:y+h, x:x+w]
            text_density = np.sum(roi > 0) / area if area > 0 else 0
            
            if 0.01 < text_density < 0.9:
                line_boxes.append((x, y, w, h))
    
    return merge_close_text_boxes(line_boxes)

def merge_overlapping_boxes(boxes):
    """Merge overlapping or close bounding boxes"""
    if not boxes:
        return []
    
    sorted_boxes = sorted(boxes, key=lambda box: box[1])
    merged_boxes = [sorted_boxes[0]]
    
    for current_box in sorted_boxes[1:]:
        prev_box = merged_boxes[-1]
        
        prev_x, prev_y, prev_w, prev_h = prev_box
        curr_x, curr_y, curr_w, curr_h = current_box
        
        prev_right = prev_x + prev_w
        prev_bottom = prev_y + prev_h
        curr_right = curr_x + curr_w
        curr_bottom = curr_y + curr_h
        
        # Check for overlap or close proximity
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

def merge_close_text_boxes(boxes, max_horizontal_gap=50, max_vertical_gap=20):
    """Merge text boxes that are close to each other"""
    if not boxes:
        return []
    
    # Group boxes by approximate vertical position
    rows = {}
    for box in boxes:
        x, y, w, h = box
        row_key = (y + h//2) // max_vertical_gap
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(box)
    
    # Merge boxes horizontally in each row
    merged_boxes = []
    for row_key, row_boxes in rows.items():
        row_boxes.sort(key=lambda box: box[0])
        
        merged_row_boxes = [row_boxes[0]]
        
        for box in row_boxes[1:]:
            last_box = merged_row_boxes[-1]
            x1, y1, w1, h1 = last_box
            x2, y2, w2, h2 = box
            
            gap = x2 - (x1 + w1)
            
            if gap <= max_horizontal_gap and abs((y1 + h1//2) - (y2 + h2//2)) <= max_vertical_gap:
                # Merge the boxes
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged_row_boxes[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged_row_boxes.append(box)
        
        merged_boxes.extend(merged_row_boxes)
    
    return merge_overlapping_boxes(merged_boxes)
    """
    Process body region to detect text elements using contour detection
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
    
    # Optional morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store bounding boxes
    boxes = []
    
    # Loop through contours to get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_width and h > min_height:  # Filter based on size
            boxes.append((x, y, w, h))
    
    # Create visualization
    img_with_boxes = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the output image with bounding boxes
    cv2.imwrite(output_path, img_with_boxes)
    
    return {
        'original': img,
        'gray': gray,
        'thresh': thresh,
        'dilated': dilated,
        'boxes': boxes,
        'visualization': img_with_boxes,
        'output_path': output_path
    }

def visualize_ocr_detection(image, detected_texts, show_confidence=True):
    """Create visualization of OCR detected text with bounding boxes"""
    img_viz = image.copy()
    
    for item in detected_texts:
        left, top, width, height = item['bbox']
        conf = item['confidence']
        text = item['text']
        
        # Color based on confidence
        if conf > 70:
            color = (0, 255, 0)  # Green for high confidence
        elif conf > 50:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw bounding box
        cv2.rectangle(img_viz, (left, top), (left + width, top + height), color, 2)
        
        # Add confidence score if requested
        if show_confidence:
            label = f"{conf:.0f}%"
            cv2.putText(img_viz, label, (left, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_viz

# Sidebar controls
st.sidebar.header("📁 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a document image", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

st.sidebar.header("🔧 Analysis Options")
show_ocr_boxes = st.sidebar.checkbox("Show OCR Text Detection", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
ocr_confidence_threshold = st.sidebar.slider("OCR Confidence Threshold", 
                                            min_value=0, max_value=100, value=30)

# Header processing parameters
st.sidebar.header("📑 Header Processing Parameters")
run_header_processing = st.sidebar.checkbox("Enable Header Layout Analysis", value=True)
logo_detection = st.sidebar.checkbox("Detect Logos", value=True)
table_detection = st.sidebar.checkbox("Detect Tables", value=True)
text_detection = st.sidebar.checkbox("Detect Text Elements", value=True)

# Body processing parameters
st.sidebar.header("📄 Body Processing Parameters")
run_body_processing = st.sidebar.checkbox("Enable Body Element Detection", value=True)
adaptive_block_size = st.sidebar.slider("Adaptive Threshold Block Size", 
                                       min_value=3, max_value=51, value=11, step=2)
adaptive_c = st.sidebar.slider("Adaptive Threshold C Value", 
                              min_value=1, max_value=20, value=5)
min_box_width = st.sidebar.slider("Minimum Box Width", 
                                 min_value=10, max_value=200, value=80)
min_box_height = st.sidebar.slider("Minimum Box Height", 
                                  min_value=5, max_value=100, value=20)

# Main processing
if uploaded_file is not None:
    # Save uploaded file
    input_path = INPUT_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Display original image
    st.subheader("📄 Original Document")
    original_image = Image.open(input_path)
    st.image(original_image, caption="Input Document", width=700)
    
    # Processing button
    if st.sidebar.button("🚀 Run Algorithmic Analysis", type="primary"):
        
        with st.spinner("Analyzing document layout... This may take a moment."):
            
            try:
                # Set up paths
                header_path = str(HEADERS_DIR / f"header_{uploaded_file.name}")
                body_path = str(BODIES_DIR / f"body_{uploaded_file.name}")
                split_path = str(SPLIT_DIR)
                
                # Run header/body detection
                results = detect_header_with_instructions_and_show_boxes(
                    str(input_path), header_path, body_path, split_path
                )
                
                st.success("✅ Layout analysis completed!")
                
                # Display results
                st.subheader("🎯 Header/Body Separation Results")
                
                # Show detection status
                col1, col2 = st.columns(2)
                with col1:
                    if results['instruction_found']:
                        st.success("🎯 'Instructions to Candidates' line detected!")
                    else:
                        st.warning("⚠️ Instructions line not found - using fallback detection")
                
                with col2:
                    st.info(f"📏 Header boundary at y = {results['header_boundary']} pixels")
                
                # Display boundary visualization
                st.subheader("📊 Boundary Detection Visualization")
                boundary_img_pil = Image.open(results['paths']['boundary'])
                st.image(boundary_img_pil, caption="Header/Body Boundary Detection", width=700)
                
                # Display separated regions
                st.subheader("📑 Separated Regions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Header Region**")
                    header_img_pil = Image.open(results['paths']['header'])
                    st.image(header_img_pil, caption="Detected Header", width=400)
                    
                    # Download button
                    with open(results['paths']['header'], "rb") as file:
                        st.download_button(
                            label="📥 Download Header",
                            data=file.read(),
                            file_name=f"header_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
                
                with col2:
                    st.markdown("**Body Region**")
                    body_img_pil = Image.open(results['paths']['body'])
                    st.image(body_img_pil, caption="Detected Body", width=400)
                    
                    # Download button
                    with open(results['paths']['body'], "rb") as file:
                        st.download_button(
                            label="📥 Download Body",
                            data=file.read(),
                            file_name=f"body_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
                
                # OCR Analysis Section
                if show_ocr_boxes:
                    st.subheader("🔍 OCR Text Detection Analysis")
                    
                    # Filter by confidence threshold
                    filtered_texts = [t for t in results['detected_texts'] 
                                    if t['confidence'] >= ocr_confidence_threshold]
                    
                    st.info(f"📊 Found {len(filtered_texts)} text elements above {ocr_confidence_threshold}% confidence")
                    
                    # Create OCR visualization
                    original_cv = cv2.imread(str(input_path))
                    ocr_viz = visualize_ocr_detection(original_cv, filtered_texts, show_confidence)
                    
                    # Convert to PIL for display
                    ocr_viz_rgb = cv2.cvtColor(ocr_viz, cv2.COLOR_BGR2RGB)
                    ocr_viz_pil = Image.fromarray(ocr_viz_rgb)
                    st.image(ocr_viz_pil, caption="OCR Text Detection (Green=High Conf, Yellow=Med, Red=Low)", width=700)
                    
                    # Show detected lines
                    with st.expander("📋 Detected Text Lines"):
                        for line_id in sorted(results['lines'].keys()):
                            line_text = ' '.join(results['lines'][line_id]['texts'])
                            st.write(f"**Line {line_id}:** {line_text}")
                
                # Header Processing Section
                if run_header_processing:
                    st.subheader("📑 Header Layout Analysis")
                    
                    with st.spinner("Analyzing header for logos, tables, and text..."):
                        
                        # Set up header processing paths
                        header_output_path = str(OUTPUT_DIR / f"header_analyzed_{uploaded_file.name}")
                        
                        # Run header layout analysis
                        header_results = analyze_document_layout(
                            results['paths']['header'],
                            header_output_path
                        )
                        
                        if header_results and header_results['detected_elements']:
                            elements = header_results['detected_elements']
                            total_elements = len(elements['logos']) + len(elements['tables']) + len(elements['text'])
                            
                            st.success(f"✅ Found {total_elements} elements in header: "
                                     f"{len(elements['logos'])} logos, "
                                     f"{len(elements['tables'])} tables, "
                                     f"{len(elements['text'])} text blocks")
                            
                            # Display header analysis results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Header Region**")
                                header_orig = Image.open(results['paths']['header'])
                                st.image(header_orig, caption="Original Header", width=500)
                            
                            with col2:
                                st.markdown("**Detected Elements**")
                                header_viz_rgb = cv2.cvtColor(header_results['result_img'], cv2.COLOR_BGR2RGB)
                                header_viz_pil = Image.fromarray(header_viz_rgb)
                                st.image(header_viz_pil, caption="Logos (Red), Tables (Blue), Text (Green)", width=500)
                            
                            # Element breakdown
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("🎨 Logos", len(elements['logos']))
                                if elements['logos']:
                                    for i, (x, y, w, h) in enumerate(elements['logos']):
                                        st.write(f"Logo {i+1}: {w}×{h} at ({x},{y})")
                            
                            with col2:
                                st.metric("📊 Tables", len(elements['tables']))
                                if elements['tables']:
                                    for i, (x, y, w, h) in enumerate(elements['tables']):
                                        st.write(f"Table {i+1}: {w}×{h} at ({x},{y})")
                            
                            with col3:
                                st.metric("📝 Text Blocks", len(elements['text']))
                                if elements['text']:
                                    total_text_area = sum(w*h for x,y,w,h in elements['text'])
                                    st.write(f"Total text area: {total_text_area:,} px²")
                            
                            # Download analyzed header
                            with open(header_output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Analyzed Header",
                                    data=file.read(),
                                    file_name=f"header_analyzed_{uploaded_file.name}",
                                    mime="image/jpeg"
                                )
                            
                            # Detailed element information
                            with st.expander("📋 Detailed Element Analysis"):
                                if elements['logos']:
                                    st.markdown("**🎨 Logo Details:**")
                                    for i, (x, y, w, h) in enumerate(elements['logos']):
                                        st.write(f"• Logo {i+1}: Position=({x},{y}), Size={w}×{h}, Area={w*h:,} px²")
                                
                                if elements['tables']:
                                    st.markdown("**📊 Table Details:**")
                                    for i, (x, y, w, h) in enumerate(elements['tables']):
                                        st.write(f"• Table {i+1}: Position=({x},{y}), Size={w}×{h}, Area={w*h:,} px²")
                                
                                if elements['text']:
                                    st.markdown("**📝 Text Block Details:**")
                                    for i, (x, y, w, h) in enumerate(elements['text']):
                                        st.write(f"• Text {i+1}: Position=({x},{y}), Size={w}×{h}, Area={w*h:,} px²")
                        
                        else:
                            st.warning("⚠️ No elements detected in header region")

                # Body Processing Section
                if run_body_processing:
                    st.subheader("📄 Body Element Detection")
                    
                    with st.spinner("Processing body region for element detection..."):
                        
                        # Set up body processing paths
                        body_output_path = str(OUTPUT_DIR / f"body_processed_{uploaded_file.name}")
                        
                        # Run body processing
                        body_results = process_body(
                            results['paths']['body'],
                            body_output_path,
                            adaptive_block_size=adaptive_block_size,
                            adaptive_c=adaptive_c,
                            min_width=min_box_width,
                            min_height=min_box_height
                        )
                        
                        if body_results:
                            st.success(f"✅ Found {len(body_results['boxes'])} text elements in body region")
                            
                            # Display processing stages
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Adaptive Threshold**")
                                thresh_pil = Image.fromarray(body_results['thresh'])
                                st.image(thresh_pil, caption="Binary Threshold", width=400)
                            
                            with col2:
                                st.markdown("**Detected Elements**")
                                viz_rgb = cv2.cvtColor(body_results['visualization'], cv2.COLOR_BGR2RGB)
                                viz_pil = Image.fromarray(viz_rgb)
                                st.image(viz_pil, caption=f"Found {len(body_results['boxes'])} Elements", width=400)
                            
                            # Element statistics
                            if body_results['boxes']:
                                box_areas = [w * h for x, y, w, h in body_results['boxes']]
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Elements", len(body_results['boxes']))
                                with col2:
                                    st.metric("Avg Area", f"{np.mean(box_areas):.0f} px²")
                                with col3:
                                    st.metric("Largest Element", f"{max(box_areas):.0f} px²")
                            
                            # Download processed body
                            with open(body_output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Processed Body",
                                    data=file.read(),
                                    file_name=f"body_processed_{uploaded_file.name}",
                                    mime="image/jpeg"
                                )
                            
                            # Show detected boxes details
                            with st.expander("📊 Detected Element Details"):
                                for i, (x, y, w, h) in enumerate(body_results['boxes']):
                                    st.write(f"**Element {i+1}:** Position=({x}, {y}), Size={w}×{h}, Area={w*h} px²")
                        
                        else:
                            st.error("❌ Body processing failed")
                
                # Complete Analysis Summary
                with st.expander("📊 Complete Analysis Summary"):
                    summary_data = {
                        "Total Text Elements (OCR)": len(results['detected_texts']),
                        "High Confidence OCR (>70%)": len([t for t in results['detected_texts'] if t['confidence'] > 70]),
                        "Medium Confidence OCR (50-70%)": len([t for t in results['detected_texts'] if 50 <= t['confidence'] <= 70]),
                        "Low Confidence OCR (<50%)": len([t for t in results['detected_texts'] if t['confidence'] < 50]),
                        "Text Lines Detected": len(results['lines']),
                        "Instructions Line Found": results['instruction_found'],
                        "Header Height (pixels)": results['header_boundary']
                    }
                    
                    # Add header analysis results
                    if run_header_processing and 'header_results' in locals() and header_results:
                        header_elements = header_results['detected_elements']
                        summary_data.update({
                            "Header - Logos Detected": len(header_elements['logos']),
                            "Header - Tables Detected": len(header_elements['tables']),
                            "Header - Text Blocks Detected": len(header_elements['text']),
                            "Header - Total Elements": len(header_elements['logos']) + len(header_elements['tables']) + len(header_elements['text'])
                        })
                    
                    # Add body analysis results
                    if run_body_processing and 'body_results' in locals() and body_results:
                        summary_data.update({
                            "Body Elements (Contour)": len(body_results['boxes']),
                            "Body Processing Parameters": f"Block={adaptive_block_size}, C={adaptive_c}, MinSize={min_box_width}×{min_box_height}"
                        })
                    
                    for key, value in summary_data.items():
                        st.write(f"**{key}:** {value}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                with st.expander("🐛 Error Details"):
                    st.code(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a document image to begin algorithmic analysis")
    
    st.subheader("📋 Algorithmic Layout Analysis Pipeline")
    st.markdown("""
    This demo showcases the **complete algorithmic approach** to document layout analysis:
    
    **Three-Stage Analysis Pipeline:**
    
    **🔍 Stage 1: Document Segmentation**
    - **OCR Text Detection** - Extract all text with bounding boxes and confidence scores
    - **Line Grouping** - Group detected text into logical lines
    - **Instructions Detection** - Search for "Instructions to Candidates" or "Candidates" text
    - **Header/Body Separation** - Split document at the instruction line with fallback heuristics
    
    **📑 Stage 2: Header Layout Analysis**
    - **Logo Detection** - Use edge density and aspect ratio analysis to find institutional logos
    - **Table Detection** - Employ line detection (horizontal + vertical) to identify tabular structures  
    - **Text Block Detection** - Use MSER + adaptive thresholding for robust text region identification
    - **Overlap Resolution** - Intelligent merging and conflict resolution between detected elements
    
    **📄 Stage 3: Body Element Detection**
    - **Adaptive Thresholding** - Binarize text regions with tunable parameters
    - **Contour Detection** - Find connected components representing text blocks
    - **Size Filtering** - Remove noise based on minimum dimensions
    - **Morphological Operations** - Connect characters and words for better detection
    
    **🎛️ Interactive Features:**
    - **Real-time parameter tuning** for all detection stages
    - **Multi-method comparison** - OCR vs Computer Vision approaches  
    - **Element-specific visualization** - Color-coded detection results (Red=Logo, Blue=Table, Green=Text)
    - **Detailed statistics** and performance metrics for each stage
    - **Downloadable outputs** for all processing stages and detected elements
    
    **🔧 Algorithmic Advantages:**
    - **No training data required** - Pure computer vision and heuristic approaches
    - **Highly interpretable** - Every detection step can be visualized and understood
    - **Fast processing** - Real-time analysis suitable for interactive demos
    - **Tunable parameters** - Easy to adapt for different document types and quality levels
    
    **Upload a document image above** to see the complete three-stage algorithmic pipeline in action!
    """)

# Cleanup function
def cleanup_temp_files():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

import atexit
atexit.register(cleanup_temp_files) #Add import for numpy at the top if not already present
import numpy as np

# Cleanup function
def cleanup_temp_files():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

import atexit
atexit.register(cleanup_temp_files)