import cv2
import os

from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, visualize_header_detection
from get_header_threshold_based import get_header_threshold_based
from get_header_hough_based import get_header_hough_based
from get_header_line_based_original import detect_header_with_instructions
from get_header_fallback import fallback_header_detection 
from display_image import display_image
from process_image import process_image


#MAIN
# Define paths
input_image_path = 'images/sample1.jpg'  
output_image_path = 'outputs/output_boxes1.jpg'

# Ensure output directory exists
if not os.path.exists('output'):
    os.makedirs('output')

header, body, boundary= detect_header_with_instructions_and_show_boxes(input_image_path)
print(f"Header/body boundary detected at y-coordinate: {boundary}")
display_image(header, "header")
display_image(body, "body")

visualize_header_detection(
    image_path=input_image_path,
    show_text_boxes=True,  # Show all text bounding boxes
    save_output=True,      # Save the output images
    output_dir="outputs"    # Save to this directory
)



# Process the image
#process_image(input_image_path, output_image_path)
