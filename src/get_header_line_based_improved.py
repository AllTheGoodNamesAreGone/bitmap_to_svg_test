import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import os
from get_header_fallback import fallback_header_detection


#improved file for line based detection - for the same line "instructions to candidates"
#also has a visualize function to see the boxes and distinction of header-body

def detect_header_with_instructions_and_show_boxes(image_path, header_path= 'headers/test.jpg', body_path = 'bodies/test.jpg', split_image_path = 'split_images'):
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
    instruction_regex_alt = re.compile (r'candidates', re.IGNORECASE)
    # Group text by line (using top coordinate and height)
    lines = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:  # Only consider text with decent confidence
            text = data['text'][i]
            if text.strip():  # Skip empty text
                top = data['top'][i]
                height = data['height'][i]
                left = data['left'][i]
                width = data['width'][i]
                
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
    
    """print("\n🔍 OCR Detected Lines:")
    for line_id in sorted(lines.keys()):
        line_text = ' '.join(lines[line_id]['texts'])
        print(f"Line {line_id}: {line_text}")"""

    # Search for instruction line in each detected line of text
    for line_id, line_data in lines.items():
        line_text = ' '.join(line_data['texts']).lower()
        if (instruction_regex.search(line_text) or instruction_regex_alt.search(line_text)):
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

    cv2.imwrite(header_path, header_img)
    cv2.imwrite(body_path, body_img)
    print(f"Header and body separated and saved to {header_path} and {body_path}")

    #drawing boundary on copy image 
    img_with_line = img.copy()
    boundary = header_boundary
    cv2.line(img_with_line, (0, boundary), (img.shape[1], boundary), (0, 0, 255), 2)
    cv2.putText(img_with_line, "Header/Body Boundary", (10, boundary - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    if split_image_path is None:
            print("Error! No output path for split images given")
            
    if not os.path.exists(split_image_path):
        os.makedirs(split_image_path)
            
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the visualization images
    cv2.imwrite(os.path.join(split_image_path, f"{base_name}_boundary.jpg"), img_with_line)
    boundary_path = os.path.join(split_image_path, f"{base_name}_boundary.jpg")

    print(f"Split image with boundary line saved to {boundary_path}")
    """cv2.imshow('Header Detection (Boundary Only)', img_with_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""                

    return header_img, body_img, header_boundary




def visualize_header_detection(image_path, show_text_boxes=True, save_output=False, output_dir=None):
    """
    Processes the image and visualizes the header detection with the 
    instructions line highlighted. Optionally shows bounding boxes for all detected text.
    
    Args:
        image_path: Path to the image file
        show_text_boxes: If True, shows bounding boxes for all detected text
        save_output: If True, saves the visualization and header/body images
        output_dir: Directory to save output images (if None, uses current directory)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Create a copy of the original image to draw on
    img_with_line = img.copy()
    img_with_boxes = img.copy()
    
    # Get header and body images
    header_img, body_img, boundary = detect_header_with_instructions_and_show_boxes(image_path)
    
    # Draw a red line at the boundary
    cv2.line(img_with_line, (0, boundary), (img.shape[1], boundary), (0, 0, 255), 2)
    cv2.line(img_with_boxes, (0, boundary), (img.shape[1], boundary), (0, 0, 255), 2)
    
    # Add text label
    cv2.putText(img_with_line, "Header/Body Boundary", (10, boundary - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_with_boxes, "Header/Body Boundary", (10, boundary - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if show_text_boxes:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to improve OCR results
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use pytesseract to extract text with bounding boxes
        custom_config = r'--oem 3 --psm 1'
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=Output.DICT)
        
        # Draw bounding boxes for all detected text
        instruction_found = False
        instruction_regex = re.compile(r'instructions\s+to\s+candidates', re.IGNORECASE)
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Only consider text with decent confidence
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                text = data['text'][i]
                
                if text.strip():  # Skip empty text
                    # Check if this text is part of "instructions to candidates"
                    if instruction_regex.search(text.lower()):
                        # Draw green box for instructions text
                        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Add the text label above the box
                        cv2.putText(img_with_boxes, "INSTRUCTIONS", (x, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        instruction_found = True
                    else:
                        # Draw blue boxes for other text
                        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)
                        
                    # Optionally display the text value (can be noisy, so commented out by default)
                    # cv2.putText(img_with_boxes, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Save output images if requested
    if save_output:
        if output_dir is None:
            output_dir = os.path.dirname(image_path) or '.'
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save the visualization images
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_boundary.jpg"), img_with_line)
        if show_text_boxes:
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_text_boxes.jpg"), img_with_boxes)
            
        # Save header and body images
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_header.jpg"), header_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_body.jpg"), body_img)
        
        print(f"Saved output images to {output_dir}")
    
    # Resize for display if too large
    max_display_height = 800
    if img_with_boxes.shape[0] > max_display_height:
        scale = max_display_height / img_with_boxes.shape[0]
        new_width = int(img_with_boxes.shape[1] * scale)
        img_with_boxes = cv2.resize(img_with_boxes, (new_width, max_display_height))
        img_with_line = cv2.resize(img_with_line, (new_width, max_display_height))
    
    # Show the images
    cv2.imshow('Header Detection (Boundary Only)', img_with_line)
    if show_text_boxes:
        cv2.imshow('Header Detection with Text Boxes', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return header_img, body_img