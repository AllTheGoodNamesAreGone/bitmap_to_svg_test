import cv2
import numpy as np
import os

def detect_tables_enhanced(color_img, gray_img):
    """
    Enhanced table detection with improved separation of individual tables
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
    
    # NEW: Improved table separation logic
    tables = find_separate_table_regions(final_h_lines, final_v_lines, width, height)
    
    # If no separate tables found, fall back to contour-based approach
    if not tables:
        tables = detect_tables_by_contours(final_h_lines, final_v_lines, width, height)
    
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

def find_separate_table_regions(h_lines, v_lines, width, height):
    """
    NEW: Improved function to find separate table regions by analyzing spatial distribution
    """
    if not h_lines or not v_lines:
        return []
    
    # Sort lines by position
    h_lines_sorted = sorted(h_lines, key=lambda x: x['y'])
    v_lines_sorted = sorted(v_lines, key=lambda x: x['x'])
    
    # Find clusters of horizontal lines (separate tables likely have gaps)
    h_clusters = find_line_clusters(h_lines_sorted, 'horizontal', gap_threshold=height//8)
    v_clusters = find_line_clusters(v_lines_sorted, 'vertical', gap_threshold=width//8)
    
    print(f"Found {len(h_clusters)} horizontal clusters and {len(v_clusters)} vertical clusters")
    
    tables = []
    
    # Create tables by combining h_clusters and v_clusters that overlap spatially
    for h_cluster in h_clusters:
        for v_cluster in v_clusters:
            # Check if horizontal and vertical line clusters overlap spatially
            if clusters_overlap(h_cluster, v_cluster):
                # Calculate table bounds
                h_min_x = min(line['x1'] for line in h_cluster)
                h_max_x = max(line['x2'] for line in h_cluster)
                h_min_y = min(line['y'] for line in h_cluster)
                h_max_y = max(line['y'] for line in h_cluster)
                
                v_min_x = min(line['x'] for line in v_cluster)
                v_max_x = max(line['x'] for line in v_cluster)
                v_min_y = min(line['y1'] for line in v_cluster)
                v_max_y = max(line['y2'] for line in v_cluster)
                
                # Use intersection of bounds to define table
                table_x = max(h_min_x, v_min_x) - 10  # Small padding
                table_y = max(h_min_y, v_min_y) - 10
                table_x2 = min(h_max_x, v_max_x) + 10
                table_y2 = min(h_max_y, v_max_y) + 10
                
                table_w = table_x2 - table_x
                table_h = table_y2 - table_y
                
                # Ensure minimum table size and reasonable aspect ratio
                if table_w > 100 and table_h > 60:
                    # Filter lines that are actually within this table region
                    table_h_lines = [line for line in h_cluster 
                                   if line['x1'] >= table_x - 20 and line['x2'] <= table_x2 + 20]
                    table_v_lines = [line for line in v_cluster 
                                   if line['y1'] >= table_y - 20 and line['y2'] <= table_y2 + 20]
                    
                    if len(table_h_lines) >= 2 and len(table_v_lines) >= 2:
                        tables.append({
                            'x': table_x, 'y': table_y,
                            'w': table_w, 'h': table_h,
                            'h_lines': table_h_lines,
                            'v_lines': table_v_lines,
                            'type': 'table'
                        })
                        print(f"Created separate table at ({table_x}, {table_y}) size {table_w}x{table_h}")
    
    # Remove overlapping tables (keep the one with more lines)
    tables = remove_overlapping_tables(tables)
    
    return tables

def find_line_clusters(lines, direction, gap_threshold):
    """
    Find clusters of lines separated by significant gaps
    """
    if not lines:
        return []
    
    clusters = []
    current_cluster = [lines[0]]
    
    for i in range(1, len(lines)):
        if direction == 'horizontal':
            prev_pos = current_cluster[-1]['y']
            curr_pos = lines[i]['y']
        else:
            prev_pos = current_cluster[-1]['x']
            curr_pos = lines[i]['x']
        
        gap = abs(curr_pos - prev_pos)
        
        if gap <= gap_threshold:
            current_cluster.append(lines[i])
        else:
            # Significant gap found - start new cluster
            if len(current_cluster) >= 2:  # Only keep clusters with multiple lines
                clusters.append(current_cluster)
            current_cluster = [lines[i]]
    
    # Don't forget the last cluster
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)
    
    return clusters

def clusters_overlap(h_cluster, v_cluster):
    """
    Check if horizontal and vertical line clusters overlap spatially
    """
    # Get bounds of horizontal cluster
    h_min_x = min(line['x1'] for line in h_cluster)
    h_max_x = max(line['x2'] for line in h_cluster)
    h_min_y = min(line['y'] for line in h_cluster)
    h_max_y = max(line['y'] for line in h_cluster)
    
    # Get bounds of vertical cluster
    v_min_x = min(line['x'] for line in v_cluster)
    v_max_x = max(line['x'] for line in v_cluster)
    v_min_y = min(line['y1'] for line in v_cluster)
    v_max_y = max(line['y2'] for line in v_cluster)
    
    # Check for overlap with some tolerance
    tolerance = 30
    
    x_overlap = not (h_max_x < v_min_x - tolerance or v_max_x < h_min_x - tolerance)
    y_overlap = not (h_max_y < v_min_y - tolerance or v_max_y < h_min_y - tolerance)
    
    return x_overlap and y_overlap

def remove_overlapping_tables(tables):
    """
    Remove overlapping tables, keeping the one with more lines
    """
    if len(tables) <= 1:
        return tables
    
    filtered_tables = []
    
    for i, table1 in enumerate(tables):
        is_overlapped = False
        
        for j, table2 in enumerate(tables):
            if i != j and tables_overlap_significantly(table1, table2):
                # Keep the table with more total lines
                lines1 = len(table1['h_lines']) + len(table1['v_lines'])
                lines2 = len(table2['h_lines']) + len(table2['v_lines'])
                
                if lines2 > lines1:
                    is_overlapped = True
                    break
        
        if not is_overlapped:
            filtered_tables.append(table1)
    
    return filtered_tables

def tables_overlap_significantly(table1, table2):
    """
    Check if two tables overlap significantly (more than 30% of smaller table)
    """
    x1, y1, w1, h1 = table1['x'], table1['y'], table1['w'], table1['h']
    x2, y2, w2, h2 = table2['x'], table2['y'], table2['w'], table2['h']
    
    # Calculate intersection
    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)
    
    if left >= right or top >= bottom:
        return False
    
    intersection_area = (right - left) * (bottom - top)
    smaller_area = min(w1 * h1, w2 * h2)
    
    overlap_ratio = intersection_area / smaller_area
    return overlap_ratio > 0.3

def detect_tables_by_contours(h_lines, v_lines, width, height):
    """
    Fallback method: Contour-based approach for table regions
    """
    tables = []
    
    # Create a mask with all detected lines
    line_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw detected lines on the mask
    for line in h_lines:
        cv2.line(line_mask, (line['x1'], line['y1']), (line['x2'], line['y2']), 255, 2)
    for line in v_lines:
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
        if area > 3000 and w > 80 and h > 40:
            # Get lines within this table region (with some padding)
            padding = 20
            table_h_lines = [line for line in h_lines 
                           if y - padding <= line['y'] <= y + h + padding]
            table_v_lines = [line for line in v_lines 
                           if x - padding <= line['x'] <= x + w + padding]
            
            # More lenient requirement for table detection
            if len(table_h_lines) >= 1 and len(table_v_lines) >= 1:
                tables.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'h_lines': table_h_lines,
                    'v_lines': table_v_lines,
                    'type': 'table'
                })
                print(f"Found table at ({x}, {y}) size {w}x{h} with {len(table_h_lines)} h-lines and {len(table_v_lines)} v-lines")
    
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

def create_table_only_svg(image_path, output_svg_path):
    """
    Create an SVG document containing only the detected table structures (lines only)
    with improved styling - black lines and thicker strokes
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    
    # Detect tables
    print("Detecting table structures...")
    tables = detect_tables_enhanced(img, gray)
    
    # Create SVG content with improved styling
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" 
     xmlns="http://www.w3.org/2000/svg">
    
    <defs>
        <style><![CDATA[
            .table-line {{ 
                stroke: #000000; 
                stroke-width: 3; 
                fill: none; 
                opacity: 1.0;
            }}
            .table-boundary {{ 
                fill: none; 
                stroke: #333333; 
                stroke-width: 2; 
                stroke-dasharray: 8,4; 
                opacity: 0.7;
            }}
            .table-label {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                font-weight: bold;
                fill: #000000;
            }}
        ]]></style>
    </defs>
    
    <!-- White background -->
    <rect width="{width}" height="{height}" fill="white"/>
    
'''
    
    if not tables:
        svg_content += '''    <!-- No tables detected -->
    <text x="50" y="50" class="table-label" fill="red">No tables detected in this image</text>
'''
    else:
        print(f"Found {len(tables)} table(s). Creating SVG...")
        
        for table_idx, table in enumerate(tables):
            table_x, table_y, table_w, table_h = table['x'], table['y'], table['w'], table['h']
            
            svg_content += f'''    <!-- Table {table_idx + 1} -->
    <g id="table_{table_idx + 1}">
        <!-- Table boundary -->
        <rect x="{table_x}" y="{table_y}" width="{table_w}" height="{table_h}" class="table-boundary"/>
        
        <!-- Table label -->
        <text x="{table_x}" y="{table_y - 8}" class="table-label">Table {table_idx + 1}</text>
        
'''
            
            # Add all lines (both horizontal and vertical) with same black styling
            h_lines = table.get('h_lines', [])
            for line_idx, h_line in enumerate(h_lines):
                x1, y1 = h_line['x1'], h_line['y1']
                x2, y2 = h_line['x2'], h_line['y2']
                svg_content += f'''        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="table-line">
            <title>Horizontal Line {line_idx + 1}</title>
        </line>
'''
            
            # Add vertical lines
            v_lines = table.get('v_lines', [])
            for line_idx, v_line in enumerate(v_lines):
                x1, y1 = v_line['x1'], v_line['y1']
                x2, y2 = v_line['x2'], v_line['y2']
                svg_content += f'''        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="table-line">
            <title>Vertical Line {line_idx + 1}</title>
        </line>
'''
            
            svg_content += f'''    </g>
    
'''
            
            print(f"Table {table_idx + 1}: {len(h_lines)} horizontal lines, {len(v_lines)} vertical lines")
    
    # Add updated legend
    svg_content += f'''    <!-- Legend -->
    <g id="legend" transform="translate(10, {height - 80})">
        <rect x="0" y="0" width="180" height="70" fill="white" stroke="black" stroke-width="1"/>
        <text x="10" y="15" class="table-label">Legend:</text>
        <line x1="10" y1="25" x2="40" y2="25" class="table-line"/>
        <text x="45" y="29" style="font-size: 12px;">Table Lines</text>
        <rect x="10" y="40" width="30" height="15" class="table-boundary"/>
        <text x="45" y="49" style="font-size: 12px;">Table Boundary</text>
    </g>
    
</svg>'''
    
    # Write SVG file
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"\nTable-only SVG created successfully at: {output_svg_path}")
    print(f"SVG dimensions: {width} x {height}")
    print(f"Total tables detected: {len(tables)}")
    
    return tables

def debug_table_detection_with_svg(image_path, output_svg_path, debug_image_path=None):
    """
    Debug table detection and create both debug image and table-only SVG
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    print("Starting enhanced table detection debug...")
    
    # Apply multiple thresholds
    binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 3)
    _, binary3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    combined_binary = cv2.bitwise_or(cv2.bitwise_or(binary1, binary2), binary3)
    
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
    
    print(f"Detected {len(final_h_lines)} horizontal lines and {len(final_v_lines)} vertical lines")
    
    # Create debug image if path provided
    if debug_image_path:
        debug_img = img.copy()
        
        # Draw detected horizontal lines in black (matching SVG style)
        for i, line in enumerate(final_h_lines):
            cv2.line(debug_img, (line['x1'], line['y1']), (line['x2'], line['y2']), (0, 0, 0), 3)
            cv2.putText(debug_img, f"H{i+1}", (line['x1'], line['y1']-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw detected vertical lines in black (matching SVG style)
        for i, line in enumerate(final_v_lines):
            cv2.line(debug_img, (line['x1'], line['y1']), (line['x2'], line['y2']), (0, 0, 0), 3)
            cv2.putText(debug_img, f"V{i+1}", (line['x1']-20, line['y1']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imwrite(debug_image_path, debug_img)
        print(f"Debug image saved at: {debug_image_path}")
    
    # Create table-only SVG
    tables = create_table_only_svg(image_path, output_svg_path)
    
    return tables, final_h_lines, final_v_lines

# Example usage:
if __name__ == "__main__":
    # Simple usage - just create table SVG
    # create_table_only_svg("question_paper.jpg", "tables_only.svg")
    
    # Debug usage - create both debug image and table SVG
    # debug_table_detection_with_svg("question_paper.jpg", "tables_only.svg", "debug_lines.jpg")
    
    pass