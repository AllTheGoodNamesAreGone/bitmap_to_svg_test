import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from pytesseract import Output
import re


def get_header (img, gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    edges = cv2.Canny(blurred, 30, 100) 
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=300, maxLineGap=10)
    debug_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10 and abs(x2 - x1) > 100:
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_image(debug_img)

    # Loop over lines and find the lowest y of a horizontal line
    header_end = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10 and abs(x2 - x1) > 100:  # horizontal line
            header_end = min(header_end, y1)

    # Now split using the first major horizontal line
    header = img[:header_end, :]
    body = img[header_end:, :]

    return header,body


def get_header_heuristic(img,gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    h, w = img.shape[:2]
    boxes = pytesseract.image_to_boxes(img)
    # Process boxes to find the header (e.g., based on y-coordinate)
    header_threshold = h * 0.4  # Define a threshold for header position
    header = []

    for box in boxes.splitlines():
        b = box.split()
        x, y, _, _ = map(int, b[1:5])
        if y < header_threshold:  # y-coordinate near the top (header)
            header.append(b)

    header_img = img[0:int(header_threshold), :]
    body_img = img[int(header_threshold):, :]

    return header_img, body_img

def detect_header_and_body(image_path):
    """
    Detects the header and body regions of a question paper without using OCR.
    Returns two images: header and body.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle varying brightness
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Invert binary image for better line detection
    binary_inv = cv2.bitwise_not(binary)
    
    # Method 1: Detect horizontal lines
    # --------------------------------------------------
    # Create horizontal kernel for line detection
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
    # --------------------------------------------------
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
    # --------------------------------------------------
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
    # --------------------------------------------------
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
        # Fallback: Use 20% of image height
        header_boundary = int(h * 0.3)
    
    # Create header and body images
    header_img = img[0:header_boundary, :]
    body_img = img[header_boundary:, :]
    
    return header_img, body_img

def detect_header_with_instructions(image_path):
    """
    Detects the header and body regions of a question paper by finding the
    "instructions to candidates" line. Returns two images: header and body.
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
    
    # Group text by line (using top coordinate and height)
    lines = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:  # Only consider text with decent confidence
            text = data['text'][i]
            if text.strip():  # Skip empty text
                top = data['top'][i]
                height = data['height'][i]
                
                # Group by line (allow for small variations in top position)
                line_id = top // 10  # Group lines within 10 pixels
                if line_id not in lines:
                    lines[line_id] = {
                        'texts': [],
                        'top': top,
                        'bottom': top + height
                    }
                lines[line_id]['texts'].append(text)
                lines[line_id]['bottom'] = max(lines[line_id]['bottom'], top + height)
    
    # Search for instruction line in each detected line of text
    for line_id, line_data in lines.items():
        line_text = ' '.join(line_data['texts']).lower()
        if instruction_regex.search(line_text):
            instruction_line_y = line_data['bottom']
            break
    
    # If we found the instruction line, use it as boundary
    if instruction_line_y:
        header_boundary = instruction_line_y
    else:
        # Fallback to the original method if no instruction line is found
        header_boundary = fallback_header_detection(gray, binary, h, w)
    
    # Create header and body images
    header_img = img[0:header_boundary, :]
    body_img = img[header_boundary:, :]
    
    return header_img, body_img, header_boundary

# Function to process image and find bounding boxes
def process_image(image_path, output_path):
    # Read the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #display_image(gray, "step 1 - gray")


    #finding header and body distinction
    header,body = get_header_heuristic(img, gray)
    display_image(header, "Header")
    display_image(body, "body")

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    #display_image(thresh, "step 2 - after thresh")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    #display_image(dilated, "step 3 - dilated")
    # Find contours (connected components)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #display_image(contours, "step 4 - contours")

    # List to store bounding boxes
    boxes = []

    # Loop through contours to get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #print(f"Box: x={x}, y={y}, w={w}, h={h}")
        if ( w > 20 and h > 20):  # Filter out noise based on box size (adjust as needed)
            boxes.append((x, y, w, h))

    # Draw bounding boxes on the original image
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    # Save the output image with bounding boxes
    display_image(img)
    cv2.imwrite(output_path, img)
    print(f"Bounding boxes drawn and saved to {output_path}")

    


def display_image(image, title="Image"):
    """Display an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()
    


#MAIN
# Define paths
input_image_path = 'images/sample1.jpg'  
output_image_path = 'outputs/output_boxes1.jpg'

# Ensure output directory exists
if not os.path.exists('output'):
    os.makedirs('output')

header, body, boundary= detect_header_with_instructions(input_image_path)
display_image(header, "header")
display_image(body, "body")

# Process the image
#process_image(input_image_path, output_image_path)
