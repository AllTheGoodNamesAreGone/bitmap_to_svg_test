import numpy as np
import cv2


def fallback_header_detection(gray, binary, h, w):
    """
    Fallback method for header detection when instructions line is not found.
    Uses a combination of horizontal lines, content density, and empty spaces.
    """
    # Invert binary image for better line detection
    binary_inv = cv2.bitwise_not(binary)
    
    # Method 1: Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.5), 1))
    horizontal_lines = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Find horizontal lines
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract line positions
    line_positions = []
    for contour in contours:
        x, y, w_line, h_line = cv2.boundingRect(contour)
        # Only consider lines that span at least 50% of the page width
        if w_line > w * 0.5:
            line_positions.append(y)
    
    # Method 2: Detect content density changes
    # Calculate horizontal projection profile (sum of pixels in each row)
    h_proj = np.sum(binary_inv, axis=1) / w  # Normalize by width
    
    # Smooth the projection profile
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size
    h_proj_smooth = np.convolve(h_proj, kernel, mode='same')
    
    # Find significant changes in density
    density_threshold = 0.1 * np.max(h_proj_smooth)
    density_changes = []
    
    for i in range(1, len(h_proj_smooth)):
        if (h_proj_smooth[i-1] < density_threshold and h_proj_smooth[i] >= density_threshold) or \
           (h_proj_smooth[i-1] >= density_threshold and h_proj_smooth[i] < density_threshold):
            density_changes.append(i)
    
    # Method 3: Find large empty spaces
    empty_threshold = 0.05 * np.max(h_proj)
    empty_regions = []
    start = None
    
    for i in range(len(h_proj)):
        if h_proj[i] < empty_threshold:
            if start is None:
                start = i
        elif start is not None:
            if i - start > 10:  # Minimum gap size (pixels)
                empty_regions.append((start, i))
            start = None
    
    if start is not None and len(h_proj) - start > 10:
        empty_regions.append((start, len(h_proj)))
    
    # Combine all methods to find header boundary
    header_candidates = []
    
    # Add horizontal lines from top 40% of page
    for pos in line_positions:
        if pos < h * 0.4:
            header_candidates.append(pos)
    
    # Add density changes from top 40% of page
    for pos in density_changes:
        if pos < h * 0.4:
            header_candidates.append(pos)
    
    # Add end of first empty region if it's in top 40%
    for start, end in empty_regions:
        if end < h * 0.4:
            header_candidates.append(end)
    
    # Determine header boundary
    if header_candidates:
        # Use the header boundary furthest down (but still in top 40%)
        header_boundary = int(max(header_candidates))
    else:
        # Fallback: Use 30% of image height
        header_boundary = int(h * 0.3)
    
    return header_boundary