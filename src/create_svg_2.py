import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import base64
from io import BytesIO
from PIL import Image, ImageFont
import re
import os

def analyze_header_layout_enhanced(header_img):
    """
    Enhanced header layout analysis with detailed text styling and structure detection
    """
    gray = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    detected_elements = {
        "text": [],
        "logos": [],
        "tables": [],
        "images": []
    }
    
    # Enhanced OCR with detailed information
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=Output.DICT)
    
    # Process text with styling information
    text_blocks = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            text = data['text'][i].strip()
            
            # Analyze text styling
            styling = analyze_text_styling(gray, x, y, w, h, text)
            
            text_blocks.append({
                'x': x, 'y': y, 'w': w, 'h': h, 
                'text': text,
                'font_size': styling['font_size'],
                'is_bold': styling['is_bold'],
                'is_italic': styling['is_italic'],
                'confidence': data['conf'][i]
            })
    
    # Group text blocks by proximity and styling
    detected_elements["text"] = group_text_blocks_enhanced(text_blocks)
    
    # Enhanced logo detection with image extraction
    detected_elements["logos"] = detect_logos_enhanced(header_img, gray)
    
    # Enhanced table detection with structure
    detected_elements["tables"] = detect_tables_enhanced(header_img, gray)
    
    return detected_elements

def analyze_text_styling(gray_image, x, y, w, h, text):
    """
    Analyze text styling (bold, italic, font size) from image region
    """
    # Extract text region
    roi = gray_image[y:y+h, x:x+w]
    
    if roi.size == 0:
        return {'font_size': 12, 'is_bold': False, 'is_italic': False}
    
    # Estimate font size based on height
    font_size = max(8, min(24, h * 0.8))
    
    # Detect bold text (thicker strokes)
    # Apply threshold to get binary image
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calculate stroke width by analyzing connected components
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    avg_thickness = 0
    if contours:
        total_area = sum(cv2.contourArea(c) for c in contours)
        total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
        if total_perimeter > 0:
            avg_thickness = total_area / total_perimeter
    
    # Bold detection based on thickness
    is_bold = avg_thickness > 1.5 or any(keyword in text.upper() for keyword in ['DEPARTMENT', 'PROGRAMME', 'INTERNAL', 'ASSESSMENT'])
    
    # Italic detection (basic slant analysis)
    is_italic = detect_italic_text(roi)
    
    return {
        'font_size': int(font_size),
        'is_bold': is_bold,
        'is_italic': is_italic
    }

def detect_italic_text(roi):
    """
    Simple italic detection based on character slant
    """
    if roi.size == 0:
        return False
    
    # Apply edge detection
    edges = cv2.Canny(roi, 50, 150)
    
    # Use Hough line detection to find dominant angles
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(roi.shape[0]*0.3))
    
    if lines is not None:
        angles = []
        for line in lines[:5]:  # Check first 5 lines
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)
        
        # If average angle deviates significantly from vertical, likely italic
        if angles:
            avg_angle = np.mean(angles)
            return abs(avg_angle - 90) > 10
    
    return False

def group_text_blocks_enhanced(text_blocks):
    """
    Enhanced text block grouping that preserves styling
    """
    if not text_blocks:
        return []
    
    # Sort by y-coordinate, then x-coordinate
    text_blocks.sort(key=lambda block: (block['y'], block['x']))
    
    grouped_blocks = []
    i = 0
    
    while i < len(text_blocks):
        current_group = [text_blocks[i]]
        j = i + 1
        
        # Group blocks on the same line
        while j < len(text_blocks):
            current_block = text_blocks[j]
            last_in_group = current_group[-1]
            
            # Check if blocks are on the same line (similar y-coordinates)
            y_diff = abs(current_block['y'] - last_in_group['y'])
            x_gap = current_block['x'] - (last_in_group['x'] + last_in_group['w'])
            
            if y_diff <= 10 and x_gap <= 50:  # Same line, reasonable gap
                current_group.append(current_block)
                j += 1
            else:
                break
        
        # Create merged block preserving the most prominent styling
        merged_block = merge_text_group(current_group)
        grouped_blocks.append(merged_block)
        
        i = j
    
    return grouped_blocks

def merge_text_group(text_group):
    """
    Merge a group of text blocks while preserving styling
    """
    if not text_group:
        return None
    
    # Calculate bounding box
    min_x = min(block['x'] for block in text_group)
    min_y = min(block['y'] for block in text_group)
    max_x = max(block['x'] + block['w'] for block in text_group)
    max_y = max(block['y'] + block['h'] for block in text_group)
    
    # Merge text with proper spacing
    sorted_blocks = sorted(text_group, key=lambda b: b['x'])
    merged_text = ' '.join(block['text'] for block in sorted_blocks)
    
    # Determine dominant styling
    font_sizes = [block['font_size'] for block in text_group]
    is_bold = any(block['is_bold'] for block in text_group)
    is_italic = any(block['is_italic'] for block in text_group)
    avg_font_size = int(np.mean(font_sizes))
    
    return {
        'x': min_x, 'y': min_y, 
        'w': max_x - min_x, 'h': max_y - min_y,
        'text': merged_text,
        'font_size': avg_font_size,
        'is_bold': is_bold,
        'is_italic': is_italic
    }

def detect_logos_enhanced(color_img, gray_img):
    """
    Enhanced logo detection with image extraction
    """
    height, width = gray_img.shape
    
    # Use multiple methods for logo detection
    logo_candidates = []
    
    # Method 1: Edge density detection
    edges = cv2.Canny(gray_img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Logo criteria: significant size, reasonable aspect ratio, high edge density
        if area > 1000 and h > 30 and w > 30 and w < width * 0.3 and h < height * 0.5:
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / area
            aspect_ratio = w / h if h > 0 else 0
            
            if edge_density > 0.03 and 0.3 < aspect_ratio < 3.0:
                # Extract logo image
                logo_img = color_img[y:y+h, x:x+w]
                logo_base64 = image_region_to_base64(logo_img)
                
                logo_candidates.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'image_data': logo_base64,
                    'edge_density': edge_density
                })
    
    # Sort by edge density and return top candidates
    logo_candidates.sort(key=lambda x: x['edge_density'], reverse=True)
    return logo_candidates[:2]  # Return top 2 logo candidates

def detect_tables_enhanced(color_img, gray_img):
    """
    Enhanced table detection with tolerance for curved/skewed documents
    """
    height, width = gray_img.shape
    
    # Multiple preprocessing approaches for better line detection
    tables = []
    
    # Method 1: Adaptive threshold with different parameters
    binary1 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    binary2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    
    # Method 2: Global threshold for darker lines
    _, binary3 = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine different threshold methods
    combined_binary = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), binary3)
    
    # More aggressive kernels for line detection
    h_kernel_sizes = [width//20, width//15, width//10]  # Multiple kernel sizes
    v_kernel_sizes = [height//20, height//15, height//10]
    
    all_h_lines = []
    all_v_lines = []
    
    # Try multiple kernel sizes to catch different line thicknesses
    for h_size in h_kernel_sizes:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        horizontal_lines = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, 
                                          horizontal_kernel, iterations=1)
        
        # Additional morphological operations to connect broken lines
        horizontal_lines = cv2.dilate(horizontal_lines, 
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), 
                                    iterations=2)
        
        h_lines = find_line_coordinates_hough(horizontal_lines, 'horizontal', width)
        all_h_lines.extend(h_lines)
    
    for v_size in v_kernel_sizes:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        vertical_lines = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, 
                                        vertical_kernel, iterations=1)
        
        # Additional morphological operations to connect broken lines
        vertical_lines = cv2.dilate(vertical_lines, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), 
                                  iterations=2)
        
        v_lines = find_line_coordinates_hough(vertical_lines, 'vertical', height)
        all_v_lines.extend(v_lines)
    
    # Remove duplicates and merge similar lines
    final_h_lines = merge_similar_lines(all_h_lines, 'horizontal', width)
    final_v_lines = merge_similar_lines(all_v_lines, 'vertical', height)
    
    print(f"Final: {len(final_h_lines)} horizontal lines and {len(final_v_lines)} vertical lines")
    
    # Method 3: Contour-based approach for table regions
    # Create a mask with all detected lines
    line_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw detected lines on the mask
    for line in final_h_lines:
        cv2.line(line_mask, (line['x1'], line['y1']), (line['x2'], line['y2']), 255, 2)
    for line in final_v_lines:
        cv2.line(line_mask, (line['x1'], line['y1']), (line['x2'], line['y2']), 255, 2)
    
    # Dilate to connect nearby lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    line_mask = cv2.dilate(line_mask, kernel, iterations=3)
    line_mask = cv2.erode(line_mask, kernel, iterations=2)
    
    # Find table regions
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # More lenient criteria for table detection
        if area > 3000 and w > 80 and h > 40:  # Reduced minimum requirements
            # Get lines within this table region (with some padding)
            padding = 20
            table_h_lines = [line for line in final_h_lines 
                           if y - padding <= line['y'] <= y + h + padding]
            table_v_lines = [line for line in final_v_lines 
                           if x - padding <= line['x'] <= x + w + padding]
            
            # More lenient requirement for table detection
            if len(table_h_lines) >= 1 and len(table_v_lines) >= 1:  # At least 1 line each direction
                tables.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'h_lines': table_h_lines,
                    'v_lines': table_v_lines,
                    'type': 'table'
                })
                print(f"Found table at ({x}, {y}) size {w}x{h} with {len(table_h_lines)} h-lines and {len(table_v_lines)} v-lines")
    
    # Fallback: If no tables found with contours, create tables from line intersections
    if not tables and len(final_h_lines) >= 2 and len(final_v_lines) >= 2:
        # Group lines that are likely part of the same table
        table_groups = find_table_regions_from_lines(final_h_lines, final_v_lines, width, height)
        tables.extend(table_groups)
    
    # Ultimate fallback: Create one large table from all lines if we have enough
    if not tables and len(final_h_lines) >= 1 and len(final_v_lines) >= 1:
        all_coords = []
        for line in final_h_lines + final_v_lines:
            all_coords.extend([line['x1'], line['y1'], line['x2'], line['y2']])
        
        if all_coords:
            min_x = min(all_coords[::2])  # x coordinates
            min_y = min(all_coords[1::2])  # y coordinates
            max_x = max(all_coords[::2])
            max_y = max(all_coords[1::2])
            
            tables.append({
                'x': min_x, 'y': min_y, 
                'w': max_x - min_x, 'h': max_y - min_y,
                'h_lines': final_h_lines,
                'v_lines': final_v_lines,
                'type': 'table'
            })
            print(f"Created fallback table with all detected lines")
    
    return tables

def find_line_coordinates_hough(line_img, direction, max_dimension):
    """
    Extract line coordinates using Hough Line Transform with tolerance for curves and skew
    """
    lines = []
    
    # Use HoughLinesP with more lenient parameters for curved/skewed lines
    detected_lines = cv2.HoughLinesP(line_img, 1, np.pi/180, threshold=30,  # Lower threshold
                                    minLineLength=max_dimension//20,  # Shorter minimum length
                                    maxLineGap=30)  # Larger gap tolerance
    
    if detected_lines is not None:
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if direction == 'horizontal':
                # For horizontal lines, allow up to 15 degrees of skew
                if abs(angle) <= 15 or abs(abs(angle) - 180) <= 15:
                    lines.append({
                        'x1': min(x1, x2), 'y1': y1,
                        'x2': max(x1, x2), 'y2': y2,
                        'y': (y1 + y2) // 2,  # Average y for sorting
                        'angle': angle
                    })
            else:  # vertical
                # For vertical lines, allow up to 15 degrees of skew
                if abs(angle - 90) <= 15 or abs(angle + 90) <= 15:
                    lines.append({
                        'x1': x1, 'y1': min(y1, y2),
                        'x2': x2, 'y2': max(y1, y2),
                        'x': (x1 + x2) // 2,  # Average x for sorting
                        'angle': angle
                    })
    
    # Group similar lines and merge them (for broken/curved lines)
    if lines:
        merged_lines = merge_similar_lines(lines, direction, max_dimension)
        return merged_lines
    
    return []

def merge_similar_lines(lines, direction, max_dimension):
    """
    Merge lines that are close together and likely part of the same table line
    """
    if not lines:
        return []
    
    # Sort lines by position
    if direction == 'horizontal':
        lines.sort(key=lambda x: x['y'])
        position_key = 'y'
        tolerance = 20  # Allow 20 pixels variation in y-position
    else:
        lines.sort(key=lambda x: x['x'])
        position_key = 'x'
        tolerance = 20  # Allow 20 pixels variation in x-position
    
    merged_lines = []
    current_group = [lines[0]]
    
    for line in lines[1:]:
        # Check if this line should be merged with the current group
        avg_position = sum(l[position_key] for l in current_group) / len(current_group)
        
        if abs(line[position_key] - avg_position) <= tolerance:
            current_group.append(line)
        else:
            # Merge the current group and start a new one
            merged_line = merge_line_group(current_group, direction)
            if merged_line:
                merged_lines.append(merged_line)
            current_group = [line]
    
    # Don't forget the last group
    merged_line = merge_line_group(current_group, direction)
    if merged_line:
        merged_lines.append(merged_line)
    
    return merged_lines

def merge_line_group(line_group, direction):
    """
    Merge a group of similar lines into one representative line
    """
    if not line_group:
        return None
    
    if direction == 'horizontal':
        # For horizontal lines, find the extent and average y-position
        min_x = min(line['x1'] for line in line_group)
        max_x = max(line['x2'] for line in line_group)
        avg_y = sum(line['y'] for line in line_group) / len(line_group)
        
        return {
            'x1': min_x, 'y1': int(avg_y),
            'x2': max_x, 'y2': int(avg_y),
            'y': int(avg_y)
        }
    else:
        # For vertical lines, find the extent and average x-position
        min_y = min(line['y1'] for line in line_group)
        max_y = max(line['y2'] for line in line_group)
        avg_x = sum(line['x'] for line in line_group) / len(line_group)
        
        return {
            'x1': int(avg_x), 'y1': min_y,
            'x2': int(avg_x), 'y2': max_y,
            'x': int(avg_x)
        }

def find_table_regions_from_lines(h_lines, v_lines, width, height):
    """
    Find table regions by analyzing line intersections and groupings
    """
    tables = []
    
    if not h_lines or not v_lines:
        return tables
    
    # Sort lines by position
    h_lines_sorted = sorted(h_lines, key=lambda x: x['y'])
    v_lines_sorted = sorted(v_lines, key=lambda x: x['x'])
    
    # Look for groups of lines that form table-like structures
    h_groups = group_nearby_lines(h_lines_sorted, 'horizontal', threshold=100)
    v_groups = group_nearby_lines(v_lines_sorted, 'vertical', threshold=100)
    
    # Create table regions from line group combinations
    for h_group in h_groups:
        for v_group in v_groups:
            if len(h_group) >= 2 and len(v_group) >= 2:
                # Calculate table bounds
                min_x = min(line['x1'] for line in v_group)
                max_x = max(line['x2'] for line in v_group)
                min_y = min(line['y1'] for line in h_group)
                max_y = max(line['y2'] for line in h_group)
                
                # Check if this forms a reasonable table
                table_w = max_x - min_x
                table_h = max_y - min_y
                
                if table_w > 50 and table_h > 30:  # Minimum table size
                    tables.append({
                        'x': min_x, 'y': min_y,
                        'w': table_w, 'h': table_h,
                        'h_lines': h_group,
                        'v_lines': v_group,
                        'type': 'table'
                    })
    
    return tables

def group_nearby_lines(lines, direction, threshold=80):
    """
    Group lines that are close together (likely part of the same table)
    """
    if not lines:
        return []
    
    groups = []
    current_group = [lines[0]]
    
    for line in lines[1:]:
        if direction == 'horizontal':
            prev_pos = current_group[-1]['y']
            curr_pos = line['y']
        else:
            prev_pos = current_group[-1]['x']
            curr_pos = line['x']
        
        if abs(curr_pos - prev_pos) <= threshold:
            current_group.append(line)
        else:
            if len(current_group) >= 1:  # Keep groups with at least 1 line
                groups.append(current_group)
            current_group = [line]
    
    # Don't forget the last group
    if len(current_group) >= 1:
        groups.append(current_group)
    
    return groups

def debug_table_detection(image_path, output_debug_path):
    """
    Enhanced debug function to visualize table detection process with curved line tolerance
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple thresholds
    binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    combined_binary = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), binary3)
    
    height, width = gray.shape
    
    # Detect lines with multiple kernel sizes
    all_h_lines = []
    all_v_lines = []
    
    h_kernel_sizes = [width//20, width//15, width//10]
    v_kernel_sizes = [height//20, height//15, height//10]
    
    for h_size in h_kernel_sizes:
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        horizontal_lines = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, 
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), 
                                    iterations=2)
        h_lines = find_line_coordinates_hough(horizontal_lines, 'horizontal', width)
        all_h_lines.extend(h_lines)
    
    for v_size in v_kernel_sizes:
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        vertical_lines = cv2.morphologyEx(combined_binary, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), 
                                  iterations=2)
        v_lines = find_line_coordinates_hough(vertical_lines, 'vertical', height)
        all_v_lines.extend(v_lines)
    
    # Merge similar lines
    final_h_lines = merge_similar_lines(all_h_lines, 'horizontal', width)
    final_v_lines = merge_similar_lines(all_v_lines, 'vertical', height)
    
    # Create debug visualization
    debug_img = img.copy()
    
    # Draw detected horizontal lines in red (with thickness to show curves)
    for i, line in enumerate(final_h_lines):
        cv2.line(debug_img, (line['x1'], line['y1']), (line['x2'], line['y2']), (0, 0, 255), 3)
        # Add line number
        cv2.putText(debug_img, f"H{i+1}", (line['x1'], line['y1']-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw detected vertical lines in blue (with thickness to show curves)
    for i, line in enumerate(final_v_lines):
        cv2.line(debug_img, (line['x1'], line['y1']), (line['x2'], line['y2']), (255, 0, 0), 3)
        # Add line number
        cv2.putText(debug_img, f"V{i+1}", (line['x1']-20, line['y1']), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Save debug image
    cv2.imwrite(output_debug_path, debug_img)
    print(f"Enhanced debug image saved at: {output_debug_path}")
    print(f"Found {len(final_h_lines)} horizontal lines and {len(final_v_lines)} vertical lines")
    print("Lines are now tolerant of up to 15 degrees of skew and can handle broken/curved lines")
    
    return final_h_lines, final_v_lines

def image_region_to_base64(img_region):
    """
    Convert image region to base64 string
    """
    try:
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img_region, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return img_base64
    except Exception as e:
        print(f"Error converting image region to base64: {e}")
        return ""

def create_content_svg(image_path, header_boundary, header_elements, output_svg_path):
    """
    Create SVG with actual document content recreation
    """
    # Read original image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    
    <defs>
        <style><![CDATA[
            .text-normal {{ font-family: Arial, sans-serif; fill: #000000; }}
            .text-bold {{ font-family: Arial, sans-serif; font-weight: bold; fill: #000000; }}
            .text-italic {{ font-family: Arial, sans-serif; font-style: italic; fill: #000000; }}
            .text-bold-italic {{ font-family: Arial, sans-serif; font-weight: bold; font-style: italic; fill: #000000; }}
            .table-line {{ stroke: #000000; stroke-width: 1; fill: none; }}
            .table-cell {{ fill: none; stroke: #cccccc; stroke-width: 0.5; }}
        ]]></style>
    </defs>
    
    <!-- Background -->
    <rect width="{width}" height="{height}" fill="white"/>
    
'''
    
    # Add text content with proper styling
    for text_block in header_elements["text"]:
        x, y, w, h = text_block['x'], text_block['y'], text_block['w'], text_block['h']
        text = escape_xml_text(text_block['text'])
        font_size = text_block['font_size']
        is_bold = text_block['is_bold']
        is_italic = text_block['is_italic']
        
        # Determine CSS class
        css_class = 'text-normal'
        if is_bold and is_italic:
            css_class = 'text-bold-italic'
        elif is_bold:
            css_class = 'text-bold'
        elif is_italic:
            css_class = 'text-italic'
        
        # Split long text into multiple lines if needed
        lines = wrap_text_to_fit(text, w, font_size)
        
        for i, line in enumerate(lines):
            text_y = y + font_size + (i * font_size * 1.2)
            svg_content += f'''    <text x="{x}" y="{text_y}" font-size="{font_size}" class="{css_class}">{line}</text>\n'''
    
    # Add embedded logos/images
    for logo in header_elements["logos"]:
        x, y, w, h = logo['x'], logo['y'], logo['w'], logo['h']
        if logo.get('image_data'):
            svg_content += f'''    <image x="{x}" y="{y}" width="{w}" height="{h}" xlink:href="data:image/png;base64,{logo['image_data']}"/>\n'''
    
    # Add table structures with proper line drawing
    for table in header_elements["tables"]:
        svg_content += f'''    <!-- Table structure -->\n'''
        
        # Draw horizontal lines
        for h_line in table.get('h_lines', []):
            x1, y1 = h_line['x1'], h_line['y1']
            x2, y2 = h_line['x2'], h_line['y2']
            svg_content += f'''    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="table-line"/>\n'''
        
        # Draw vertical lines  
        for v_line in table.get('v_lines', []):
            x1, y1 = v_line['x1'], v_line['y1']
            x2, y2 = v_line['x2'], v_line['y2']
            svg_content += f'''    <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="table-line"/>\n'''
        
        # Optional: Draw table boundary
        table_x, table_y, table_w, table_h = table['x'], table['y'], table['w'], table['h']
        svg_content += f'''    <rect x="{table_x}" y="{table_y}" width="{table_w}" height="{table_h}" fill="none" stroke="#ff0000" stroke-width="2" stroke-dasharray="5,5"/>\n'''
    
    svg_content += '</svg>'
    
    # Write SVG file
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"Content SVG created at: {output_svg_path}")

def wrap_text_to_fit(text, max_width, font_size):
    """
    Simple text wrapping based on estimated character width
    """
    # Estimate character width (rough approximation)
    char_width = font_size * 0.6
    max_chars = int(max_width / char_width)
    
    if len(text) <= max_chars:
        return [text]
    
    # Simple word wrapping
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= max_chars:
            current_line += (" " if current_line else "") + word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

def escape_xml_text(text):
    """
    Escape special XML characters
    """
    if not text:
        return ""
    
    text = str(text)
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    
    return text

# Modified main function with debug option
def generate_content_svg(image_path, output_svg_path, debug_tables=True):
    """
    Generate SVG with actual document content
    """
    # Import your existing header detection function
    from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes
    
    # Step 1: Split the document
    header_img, body_img, header_boundary = detect_header_with_instructions_and_show_boxes(
        image_path, 
        header_path='temp_header.jpg', 
        body_path='temp_body.jpg', 
        split_image_path='temp_split'
    )
    
    # Debug table detection if requested
    if debug_tables:
        print("Running table detection debug...")
        h_lines, v_lines = debug_table_detection(image_path, "debug_table_detection.jpg")
    
    # Step 2: Enhanced analysis
    header_elements = analyze_header_layout_enhanced(header_img)
    
    # Print detection results
    print(f"\nDetection Results:")
    print(f"- Text blocks: {len(header_elements['text'])}")
    print(f"- Logos: {len(header_elements['logos'])}")
    print(f"- Tables: {len(header_elements['tables'])}")
    
    for i, table in enumerate(header_elements['tables']):
        print(f"  Table {i+1}: {len(table.get('h_lines', []))} h-lines, {len(table.get('v_lines', []))} v-lines")
    
    # Step 3: Create content SVG
    create_content_svg(image_path, header_boundary, header_elements, output_svg_path)
    
    # Clean up
    for temp_file in ['temp_header.jpg', 'temp_body.jpg']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print(f"Content SVG generated at: {output_svg_path}")

# Example usage with debug:
# generate_content_svg("question_paper.jpg", "document_content.svg", debug_tables=True)