import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from scipy.ndimage import binary_fill_holes
from PIL import Image
import pytesseract  # For OCR-based layout analysis (optional)
from display_image import display_image

def analyze_document_layout(image_path, output_path, display_steps=False):
    """
    Advanced document layout analysis that identifies and draws bounding boxes around
    text paragraphs, tables, logos and other document elements.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the output image with bounding boxes
        display_steps (bool): Whether to display intermediate processing steps
    
    Returns:
        tuple: (Output image with bounding boxes, dict of detected elements)
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
        "paragraphs": [],
        "tables": [],
        "logos": [],
        "headers_footers": [],
        "other_elements": []
    }
    
    # Step 1: Detect and extract the logo first (typically in top left/right corner)
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
    filtered_table_boxes = []
    for box in table_boxes:
        x, y, w, h = box
        table_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(table_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(table_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h)
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            filtered_table_boxes.append(box)
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["tables"].append(box)
            # Draw blue bounding box around table
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_img, "Table", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Step 3: Detect text paragraphs
    paragraph_boxes = detect_paragraphs(gray, height, width)
    
    # Filter out paragraphs that overlap with already detected elements
    filtered_paragraph_boxes = []
    for box in paragraph_boxes:
        x, y, w, h = box
        para_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(para_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(para_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h)
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            filtered_paragraph_boxes.append(box)
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["paragraphs"].append(box)
            # Draw green bounding box around paragraph
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_img, "Paragraph", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Step 4: Detect headers and footers (typically at top and bottom of document)
    header_footer_boxes = detect_headers_footers(gray, height, width)
    
    # Filter and add headers/footers
    for box in header_footer_boxes:
        x, y, w, h = box
        hf_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(hf_mask, (x, y), (x+w, y+h), 255, -1)
        
        overlap = cv2.bitwise_and(hf_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h)
        
        if overlap_percentage < 0.3:
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["headers_footers"].append(box)
            # Draw orange bounding box around header/footer
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 165, 255), 2)
            element_type = "Header" if y < height/3 else "Footer"
            cv2.putText(result_img, element_type, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # Save the result
    cv2.imwrite(output_path, result_img)
    print(f"Layout analysis complete. Result saved to {output_path}")
    
    if display_steps:
        display_image(result_img, "Final Result with Layout Analysis")
    
    return result_img, detected_elements

def detect_logo(gray_image, height, width):
    """
    Detect logo in the document image (typically in top portion)
    """
    # Focus on top portion of the document
    top_portion = height // 4
    gray_top = gray_image[:top_portion, :]
    
    # Use edge detection to find regions with high edge density (likely logo)
    edges = cv2.Canny(gray_top, 50, 150)
    
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
        
        # Logo is typically in the top left or right corner with significant edge density
        if area > 3000 and h < top_portion and w > 50:
            # Check if it's in left or right portion
            if x < width // 3 or x > 2*width//3:
                # Check edge density inside the region
                roi = edges[y:y+h, x:x+w]
                edge_density = np.sum(roi > 0) / area
                
                if edge_density > 0.05:  # Adjust threshold as needed
                    logo_boxes.append((x, y, w, h))
    
    # Merge overlapping boxes
    logo_boxes = merge_overlapping_boxes(logo_boxes)
    
    # If multiple logos detected, keep the largest one or the one in the left corner
    if len(logo_boxes) > 1:
        logo_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        logo_boxes = [logo_boxes[0]]
    
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

def detect_paragraphs(gray_image, height, width):
    """
    Detect text paragraphs using connected component analysis and filtering
    """
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Connect characters within words and words within lines
    kernel_connect_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    text_mask = cv2.dilate(binary, kernel_connect_horizontal, iterations=1)
    
    # Connect lines within paragraphs (with smaller vertical dilation)
    kernel_connect_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    text_mask = cv2.dilate(text_mask, kernel_connect_vertical, iterations=1)
    
    # Apply closing to fill gaps
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, 
                               cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5)), 
                               iterations=1)
    
    # Find contours for potential paragraphs
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and shape characteristics
    paragraph_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Paragraphs typically have width greater than height and reasonable size
        if area > 5000 and w > 80 and h > 20:
            # Check text density to filter out non-text regions
            roi = binary[y:y+h, x:x+w]
            text_density = np.sum(roi > 0) / area
            
            # Paragraphs have moderate text density
            if 0.05 < text_density < 0.5:
                paragraph_boxes.append((x, y, w, h))
    
    # Add an additional check for multi-line paragraphs (horizontal stripes pattern)
    refined_paragraph_boxes = []
    for box in paragraph_boxes:
        x, y, w, h = box
        roi = binary[y:y+h, x:x+w]
        
        # Project text horizontally to detect text lines
        h_projection = np.sum(roi > 0, axis=1)
        
        # Count transitions between text and non-text (indicating multiple lines)
        transitions = 0
        prev = 0
        for val in h_projection:
            if prev == 0 and val > 0:
                transitions += 1
            prev = 1 if val > 0 else 0
        
        # Multi-line paragraphs have multiple transitions
        if transitions >= 2:
            refined_paragraph_boxes.append(box)
        # For single-line text, apply additional criteria
        elif w > 200 and aspect_ratio > 3:
            refined_paragraph_boxes.append(box)
    
    # Merge overlapping boxes
    paragraph_boxes = merge_overlapping_boxes(refined_paragraph_boxes)
    
    return paragraph_boxes

def detect_headers_footers(gray_image, height, width):
    """
    Detect headers and footers in the document
    """
    # Focus on top and bottom portions of the document
    header_region = gray_image[:height//6, :]
    footer_region = gray_image[5*height//6:, :]
    
    # Process header
    header_binary = cv2.adaptiveThreshold(header_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 15)
    
    # Process footer
    footer_binary = cv2.adaptiveThreshold(footer_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 15)
    
    # Connect components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    header_mask = cv2.dilate(header_binary, kernel, iterations=1)
    footer_mask = cv2.dilate(footer_binary, kernel, iterations=1)
    
    # Find contours
    header_contours, _ = cv2.findContours(header_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    footer_contours, _ = cv2.findContours(footer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process header contours
    header_boxes = []
    for contour in header_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > 3000 and w > 100:
            # Adjust y-coordinate to match the original image
            header_boxes.append((x, y, w, h))
    
    # Process footer contours
    footer_boxes = []
    for contour in footer_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        if area > 3000 and w > 100:
            # Adjust y-coordinate to match the original image
            footer_boxes.append((x, 5*height//6 + y, w, h))
    
    # Merge overlapping boxes
    header_boxes = merge_overlapping_boxes(header_boxes)
    footer_boxes = merge_overlapping_boxes(footer_boxes)
    
    return header_boxes + footer_boxes

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

