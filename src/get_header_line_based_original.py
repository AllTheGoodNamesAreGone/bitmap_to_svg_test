import cv2
import pytesseract
from get_header_fallback import fallback_header_detection
import re
from pytesseract import Output


#Finds the line "Instructions to candidates" - case insensitive, and sets the header to end just below that line 
#original function for this, does not print boxes or anything, works fine! it finds the line well, will try for more pics


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