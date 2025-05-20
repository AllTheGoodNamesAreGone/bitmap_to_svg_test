import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
from scipy.ndimage import binary_fill_holes
from display_image import display_image

#HEADER PROCESSING 

def analyze_document_layout(image_path, output_path, display_steps=False):
    """
    Enhanced document layout analysis that identifies and draws bounding boxes around
    text (paragraphs/lines), logos, and tables with improved text detection.
    
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
        "text": [],
        "logos": [],
        "tables": []
    }
    
    # Step 1: Detect and extract the logo (can be anywhere in the document)
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
    
    # Step 3: Detect text (paragraphs and lines) with improved parameters
    text_boxes = detect_text_improved(gray, height, width, mask_detected)
    
    # Filter out text regions that overlap with already detected elements
    filtered_text_boxes = []
    for box in text_boxes:
        x, y, w, h = box
        text_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(text_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Check overlap percentage
        overlap = cv2.bitwise_and(text_mask, mask_detected)
        overlap_percentage = np.sum(overlap > 0) / (w * h)
        
        if overlap_percentage < 0.3:  # Less than 30% overlap
            filtered_text_boxes.append(box)
            cv2.rectangle(mask_detected, (x, y), (x+w, y+h), 255, -1)
            detected_elements["text"].append(box)
            # Draw green bounding box around text
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_img, "Text", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite(output_path, result_img)
    print(f"Layout analysis complete. Result saved to {output_path}")
    
    if display_steps:
        plt.figure(figsize=(10, 10))
        plt.title("Final Result with Layout Analysis")
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return result_img, detected_elements

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

def detect_text_improved(gray_image, height, width, existing_mask=None):
    """
    Improved version of text detection with better parameters for document headers
    """
    # Apply adaptive thresholding with parameters tuned for text
    binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # If there's an existing mask, exclude those regions
    if existing_mask is not None:
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(existing_mask))
    
    # Create connected components for text analysis using MSER
    # (Maximally Stable Extremal Regions) - good for text detection
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray_image)
    
    # Create mask for MSER regions
    mser_mask = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(mser_mask, [hull], 0, 255, -1)
    
    # Combine with binary image for robust text detection
    text_mask = cv2.bitwise_or(binary, mser_mask)
    
    # Connect characters within words horizontally
    kernel_connect_chars = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    word_mask = cv2.dilate(text_mask, kernel_connect_chars, iterations=1)
    
    # Connect words vertically with a smaller dilation to maintain text line separation
    kernel_connect_words = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    line_mask = cv2.dilate(word_mask, kernel_connect_words, iterations=1)
    
    # Find contours for text lines
    contours, _ = cv2.findContours(line_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for individual lines
    line_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filter based on reasonable text line dimensions
        if area > 500 and w > 20 and h > 8:
            # Check text density
            roi = text_mask[y:y+h, x:x+w]
            text_density = np.sum(roi > 0) / area
            
            # Text has moderate density
            if 0.01 < text_density < 0.9:
                line_boxes.append((x, y, w, h))
    
    # Connect text lines into paragraphs
    paragraph_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw detected lines on the mask
    for x, y, w, h in line_boxes:
        cv2.rectangle(paragraph_mask, (x, y), (x+w, y+h), 255, -1)
    
    # Connect nearby lines into paragraphs - adjusted for headers with larger text
    kernel_connect_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))  # Connect vertically
    paragraph_mask = cv2.dilate(paragraph_mask, kernel_connect_lines, iterations=1)
    
    # Also connect horizontally for fragmented text in the same line
    kernel_connect_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))  # Connect horizontally
    paragraph_mask = cv2.dilate(paragraph_mask, kernel_connect_horizontal, iterations=1)
    
    # Find contours for paragraphs
    paragraph_contours, _ = cv2.findContours(paragraph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for paragraphs
    paragraph_boxes = []
    for contour in paragraph_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Paragraphs have more substantial size
        if area > 500:  # Lower threshold to catch smaller text blocks
            paragraph_boxes.append((x, y, w, h))
    
    # Merge close text regions (specific to document headers)
    text_boxes = merge_close_text_boxes(paragraph_boxes, max_horizontal_gap=100, max_vertical_gap=15)
    
    return text_boxes

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

def merge_close_text_boxes(boxes, max_horizontal_gap=50, max_vertical_gap=20):
    """
    Merge text boxes that are close to each other (specifically for document headers)
    
    Args:
        boxes: List of tuples (x, y, w, h)
        max_horizontal_gap: Maximum horizontal gap between boxes to be merged
        max_vertical_gap: Maximum vertical gap between boxes to be merged
        
    Returns:
        List of merged boxes
    """
    if not boxes:
        return []
    
    # Step 1: Group boxes by their approximate vertical position (row-wise)
    rows = {}
    for box in boxes:
        x, y, w, h = box
        # Use the center y coordinate as a key, rounded to nearest 10 pixels as tolerance
        row_key = (y + h//2) // max_vertical_gap
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(box)
    
    # Step 2: For each row, merge boxes that are horizontally close
    merged_boxes = []
    for row_key, row_boxes in rows.items():
        # Sort boxes horizontally
        row_boxes.sort(key=lambda box: box[0])
        
        # Start with the first box in the row
        merged_row_boxes = [row_boxes[0]]
        
        # Try to merge with subsequent boxes
        for box in row_boxes[1:]:
            last_box = merged_row_boxes[-1]
            x1, y1, w1, h1 = last_box
            x2, y2, w2, h2 = box
            
            # Calculate horizontal gap
            gap = x2 - (x1 + w1)
            
            # If boxes are close horizontally and approximately at the same height
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
    
    # Step 3: Final pass to merge any overlapping boxes created in the process
    return merge_overlapping_boxes(merged_boxes)

"""
def display_image(image, title="Image"):
    
    Display an image using matplotlib
    
    Args:
        image: Image to display
        title: Title for the image window
    
    plt.figure(figsize=(10, 10))
    plt.title(title)
    
    if len(image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        plt.imshow(image, cmap='gray')
        
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    None

"""