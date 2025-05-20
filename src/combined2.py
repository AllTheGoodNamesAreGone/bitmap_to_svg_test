import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

def process_exam_paper(image_path, output_svg_path):
    """
    Process an exam paper image and convert it to SVG format
    
    Args:
        image_path (str): Path to the input exam image
        output_svg_path (str): Path to save the output SVG
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read the image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None
    
    # Get image dimensions
    height, width = original_img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # Process each section of the exam paper
    result_img, sections = segment_exam_paper(original_img, gray)
    
    # Create SVG from sections
    create_svg_from_sections(sections, width, height, output_svg_path)
    
    # Save the debug image with regions
    debug_output_path = output_svg_path.replace('.svg', '_debug.jpg')
    cv2.imwrite(debug_output_path, result_img)
    
    print(f"Exam paper processed. SVG saved to {output_svg_path}")
    print(f"Debug image saved to {debug_output_path}")
    
    return result_img, sections

def segment_exam_paper(original_img, gray):
    """
    Segment the exam paper into different sections:
    - Header with logo
    - Course information
    - Instructions
    - Questions table
    
    Returns:
        tuple: (Annotated image, dictionary of detected sections)
    """
    height, width = gray.shape
    result_img = original_img.copy()
    
    # Dictionary to store detected elements
    sections = {
        "logo": None,
        "header_text": [],
        "course_info": [],
        "instructions": [],
        "questions_table": []
    }
    
    # Detect logo (usually in the top-left corner)
    logo_box = detect_logo(gray, height, width)
    if logo_box:
        x, y, w, h = logo_box
        sections["logo"] = (x, y, w, h)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result_img, "Logo", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Detect header text (title, department, etc.)
    header_boxes = detect_header_text(gray, height, width, logo_box)
    for i, (x, y, w, h) in enumerate(header_boxes):
        sections["header_text"].append((x, y, w, h))
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(result_img, f"Header {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Detect tables (course info table, questions table)
    table_boxes = detect_tables(gray, height, width)
    
    # First table is usually the course info
    if table_boxes:
        x, y, w, h = table_boxes[0]
        sections["course_info"] = (x, y, w, h)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_img, "Course Info", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Last and largest table is usually the questions table
    if len(table_boxes) > 1:
        # Sort by area (descending)
        table_boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        x, y, w, h = table_boxes[0]  # Largest table
        sections["questions_table"] = (x, y, w, h)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(result_img, "Questions", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Detect instructions text (usually between course info and questions)
    instruction_boxes = detect_instructions(gray, height, width, sections)
    for i, (x, y, w, h) in enumerate(instruction_boxes):
        sections["instructions"].append((x, y, w, h))
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(result_img, f"Instruction {i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    return result_img, sections

def detect_logo(gray_image, height, width):
    """
    Detect logo in the document image (usually top-left corner)
    """
    # Focus on the top-left corner for logo
    top_height = int(height * 0.15)  # Top 15% of the image
    left_width = int(width * 0.3)    # Left 30% of the image
    
    top_left_region = gray_image[:top_height, :left_width]
    
    # Apply adaptive thresholding to isolate potential logo components
    binary = cv2.adaptiveThreshold(top_left_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 15, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size
    logo_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Logo typically has significant size
        if area > 1000:
            logo_contours.append(contour)
    
    # If we found logo contours, find their bounding box
    if logo_contours:
        # Find the combined bounding box
        min_x, min_y = width, height
        max_x, max_y = 0, 0
        
        for contour in logo_contours:
            x, y, w, h = cv2.boundingRect(contour)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        # Return the logo bounding box
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    # Default logo position if not detected
    return (10, 10, left_width - 20, top_height - 20)

def detect_header_text(gray_image, height, width, logo_box=None):
    """
    Detect header text (title, department info, etc.)
    """
    # Focus on the top portion for header text
    top_height = int(height * 0.25)  # Top 25% of the image
    header_region = gray_image[:top_height, :]
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(header_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    
    # Exclude logo region if provided
    if logo_box:
        x, y, w, h = logo_box
        cv2.rectangle(binary, (x, y), (x+w, y+h), 0, -1)
    
    # Connect characters horizontally (for text lines)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for text lines
    text_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Text lines typically have reasonable width-to-height ratio
        aspect_ratio = w / float(h) if h > 0 else 0
        
        if area > 1000 and aspect_ratio > 3:  # Text lines are usually wide
            text_boxes.append((x, y, w, h))
    
    # Sort boxes vertically (top to bottom)
    text_boxes.sort(key=lambda box: box[1])
    
    return text_boxes

def detect_tables(gray_image, height, width):
    """
    Detect tables in the exam paper (course info table, questions table)
    """
    # Detect horizontal and vertical lines
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
    
    # Find contours for tables
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for tables
    table_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Tables typically have significant size
        if area > 20000:  # Adjust based on your image size
            table_boxes.append((x, y, w, h))
    
    # Sort tables by y-coordinate (top to bottom)
    table_boxes.sort(key=lambda box: box[1])
    
    return table_boxes

def detect_instructions(gray_image, height, width, sections):
    """
    Detect instruction text (usually between course info and questions table)
    """
    # Determine the region to look for instructions
    start_y = 0
    end_y = height
    
    if "course_info" in sections and sections["course_info"]:
        x, y, w, h = sections["course_info"]
        start_y = y + h + 10  # Just below course info table
    
    if "questions_table" in sections and sections["questions_table"]:
        x, y, w, h = sections["questions_table"]
        end_y = y - 10  # Just above questions table
    
    # If the section doesn't make sense, return empty
    if start_y >= end_y:
        return []
    
    # Extract the region
    instruction_region = gray_image[start_y:end_y, :]
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(instruction_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    
    # Connect characters horizontally (for text lines)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours for text lines
    instruction_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y += start_y  # Adjust y-coordinate back to original image
        area = w * h
        
        # Instruction text typically spans a good portion of width
        if area > 5000 and w > width * 0.3:
            instruction_boxes.append((x, y, w, h))
    
    # Sort boxes vertically (top to bottom)
    instruction_boxes.sort(key=lambda box: box[1])
    
    return instruction_boxes

def create_svg_from_sections(sections, width, height, output_path):
    """
    Create an SVG representation of the exam paper based on detected sections
    """
    # Create SVG root element
    svg = Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'width': str(width),
        'height': str(height),
        'viewBox': f'0 0 {width} {height}'
    })
    
    # Add a white background
    SubElement(svg, 'rect', {
        'width': str(width),
        'height': str(height),
        'fill': 'white'
    })
    
    # Helper function to create a box
    def add_box(x, y, w, h, color, stroke_width, label=None):
        box = SubElement(svg, 'rect', {
            'x': str(x),
            'y': str(y),
            'width': str(w),
            'height': str(h),
            'stroke': color,
            'stroke-width': str(stroke_width),
            'fill': 'none'
        })
        
        if label:
            text = SubElement(svg, 'text', {
                'x': str(x + 5),
                'y': str(y - 5),
                'font-family': 'Arial',
                'font-size': '12',
                'fill': color
            })
            text.text = label
    
    # Draw logo area
    if sections["logo"]:
        x, y, w, h = sections["logo"]
        add_box(x, y, w, h, 'red', 2, 'Logo')
        
        # Add placeholder logo shape
        logo_group = SubElement(svg, 'g', {
            'transform': f'translate({x+5}, {y+5})'
        })
        SubElement(logo_group, 'circle', {
            'cx': '20',
            'cy': '20',
            'r': '15',
            'fill': '#eee',
            'stroke': 'red',
            'stroke-width': '1'
        })
    
    # Draw header text areas
    for i, (x, y, w, h) in enumerate(sections["header_text"]):
        add_box(x, y, w, h, 'blue', 2, f'Header {i+1}')
        
        # Add placeholder text line
        text = SubElement(svg, 'text', {
            'x': str(x + 10),
            'y': str(y + h/2 + 5),
            'font-family': 'Arial',
            'font-size': '14',
            'fill': '#333'
        })
        text.text = f'Header Text {i+1}'
    
    # Draw course info table
    if "course_info" in sections and sections["course_info"]:
        x, y, w, h = sections["course_info"]
        add_box(x, y, w, h, 'green', 2, 'Course Info')
        
        # Add simple table structure (3x2 grid)
        cell_width = w / 2
        cell_height = h / 3
        for row in range(3):
            for col in range(2):
                cell_x = x + col * cell_width
                cell_y = y + row * cell_height
                SubElement(svg, 'rect', {
                    'x': str(cell_x),
                    'y': str(cell_y),
                    'width': str(cell_width),
                    'height': str(cell_height),
                    'stroke': 'green',
                    'stroke-width': '1',
                    'fill': 'none'
                })
    
    # Draw instructions
    for i, (x, y, w, h) in enumerate(sections["instructions"]):
        add_box(x, y, w, h, 'purple', 2, f'Instruction {i+1}')
        
        # Add placeholder text line
        text = SubElement(svg, 'text', {
            'x': str(x + 10),
            'y': str(y + h/2 + 5),
            'font-family': 'Arial',
            'font-size': '14',
            'fill': '#333'
        })
        text.text = 'Instructions to candidates'
    
    # Draw questions table
    if "questions_table" in sections and sections["questions_table"]:
        x, y, w, h = sections["questions_table"]
        add_box(x, y, w, h, 'orange', 2, 'Questions Table')
        
        # Add simple table structure
        num_rows = 6  # Approximate number of rows
        num_cols = 4  # Q.No, Question, CO, Marks
        
        cell_width = w / num_cols
        cell_height = h / num_rows
        
        # Draw grid
        for row in range(num_rows):
            for col in range(num_cols):
                cell_x = x + col * cell_width
                cell_y = y + row * cell_height
                SubElement(svg, 'rect', {
                    'x': str(cell_x),
                    'y': str(cell_y),
                    'width': str(cell_width),
                    'height': str(cell_height),
                    'stroke': 'orange',
                    'stroke-width': '1',
                    'fill': 'none'
                })
        
        # Add headers
        header_texts = ['Q.No', 'Questions', 'CO', 'Marks']
        for col, header in enumerate(header_texts):
            text = SubElement(svg, 'text', {
                'x': str(x + col * cell_width + 10),
                'y': str(y + 20),
                'font-family': 'Arial',
                'font-size': '14',
                'fill': '#333'
            })
            text.text = header
    
    # Convert to pretty XML
    rough_string = tostring(svg, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_svg = reparsed.toprettyxml(indent="  ")
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(pretty_svg)

def process_exam_paper_batch(input_folder, output_folder):
    """
    Process multiple exam papers in a folder
    
    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path to folder for output SVGs
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    # Process each file
    for file in image_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.rsplit('.', 1)[0] + '.svg')
        
        print(f"Processing {file}...")
        try:
            process_exam_paper(input_path, output_path)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    print(f"Batch processing complete. {len(image_files)} files processed.")

def display_image(image, title="Image"):
    """
    Display an image using matplotlib
    
    Args:
        image: Image to display
        title: Title for the image window
    """
    plt.figure(figsize=(10, 10))
    plt.title(title)
    
    if len(image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale image
        plt.imshow(image, cmap='gray')
        
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Usage examples
if __name__ == "__main__":
    # Process a single file
    # image_path = "exam_paper.jpg"
    # output_path = "exam_paper.svg"
    # process_exam_paper(image_path, output_path)
    
    # Process multiple files in a directory
    # input_folder = "exam_papers"
    # output_folder = "exam_papers_svg"
    # process_exam_paper_batch(input_folder, output_folder)
    
    print("Script ready. Please call functions with appropriate parameters.")