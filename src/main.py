import cv2
import os

from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, visualize_header_detection
from get_header_threshold_based import get_header_threshold_based
from get_header_hough_based import get_header_hough_based
from get_header_line_based_original import detect_header_with_instructions
from get_header_fallback import fallback_header_detection 
from display_image import display_image
from process_image import process_image


# Directories
input_dir = 'images'
output_dir = 'outputs'
header_dir = 'headers'
body_dir = 'bodies'
split_dir = 'split_images'

# Create output directories if not exist
for directory in [output_dir, header_dir, body_dir, split_dir]:
    os.makedirs(directory, exist_ok=True)

# Supported image formats
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
already_processed = 0
newly_processed = 0
# Process each image
for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(input_dir, filename)

        # Output paths using the same base filename
        base_name = os.path.splitext(filename)[0]
        output_image_path = os.path.join(output_dir, f"{base_name}_output.jpg")
        header_path = os.path.join(header_dir, f"{base_name}_header.jpg")
        body_path = os.path.join(body_dir, f"{base_name}_body.jpg")
        split_image_path = split_dir

        # Skip if output already exists
        if os.path.exists(header_path) and os.path.exists(body_path):
            print(f"✅ Skipping {filename} (already processed).")
            already_processed = already_processed +1
            continue

        try:
            print(f"\n🚀 Processing {filename}...")
            header, body, boundary = detect_header_with_instructions_and_show_boxes(
                image_path, header_path, body_path, split_image_path
            )
            print(f"Header/body boundary for {filename} at y = {boundary}")
            newly_processed = newly_processed +1
            # Optional: display the result visually (can be commented out for batch runs)
            #display_image(header, f"{filename} - header")
            #display_image(body, f"{filename} - body")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print (f"{already_processed} files left untouched, {newly_processed} files processed")

visualize_header_detection('images/sample2.jpg')

"""
#MAIN
# Define paths
input_image_path = 'images/sample1.jpg'  
output_image_path = 'outputs/output_boxes1.jpg'
header_path = 'headers/header1.jpg'
body_path = 'bodies/body1.jpg'
split_image_path = 'split_images'

# Ensure output directory exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

header, body, boundary= detect_header_with_instructions_and_show_boxes(input_image_path, header_path, body_path, split_image_path)
print(f"Header/body boundary detected at y-coordinate: {boundary}")
#display_image(header, "header")
#display_image(body, "body")
"""
"""visualize_header_detection(
    image_path=input_image_path,
    show_text_boxes=True,  # Show all text bounding boxes
    save_output=True,      # Save the output images
    output_dir="outputs"    # Save to this directory
)"""



# Process the image
#process_image(input_image_path, output_image_path)
