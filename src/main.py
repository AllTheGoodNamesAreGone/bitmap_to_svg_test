import cv2
import os

from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, visualize_header_detection
from get_header_threshold_based import get_header_threshold_based
from get_header_hough_based import get_header_hough_based
from get_header_line_based_original import detect_header_with_instructions
from get_header_fallback import fallback_header_detection 
from display_image import display_image
from process_image import process_image, process_body
#from layout_analyzer import analyze_document_layout, detect_logo, detect_headers_footers, detect_paragraphs, detect_tables, merge_overlapping_boxes
#from layout_analyzer_v2 import analyze_document_layout, detect_logo, detect_tables, detect_text, merge_overlapping_boxes
from layout_analyzer_v3 import analyze_document_layout, detect_logo, detect_tables, detect_text_improved, merge_close_text_boxes, merge_overlapping_boxes


# Directories
input_dir = 'images'
output_body_dir = 'outputs/body_outputs/trial1'
output_header_dir = 'outputs/header_outputs/trial1'
header_dir = 'headers'
body_dir = 'bodies'
split_dir = 'split_images'

# Create output directories if not exist
for directory in [output_body_dir, header_dir, body_dir, split_dir]:
    os.makedirs(directory, exist_ok=True)

# Supported image formats
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
already_processed = 0
newly_processed = 0

# Process each image input file
for filename in os.listdir(input_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(input_dir, filename)

        # Output paths using the same base filename
        base_name = os.path.splitext(filename)[0]
        #output_image_path = os.path.join(output_body_dir, f"{base_name}_output.jpg")
        header_path = os.path.join(header_dir, f"{base_name}_header.jpg")
        body_path = os.path.join(body_dir, f"{base_name}_body.jpg")
        split_image_path = split_dir

        # Skip if output already exists
        if (os.path.exists(header_path) and os.path.exists(body_path)) or (filename.lower().endswith("part2.jpg")) :
            #print(f"✅ Skipping {filename} (already processed).")
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

print (f"{already_processed} image file(s) left untouched, {newly_processed} image file(s) processed")


#Process each body file
bodies_already_processed = 0
bodies_newly_processed =0
for filename in os.listdir(body_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(body_dir, filename)

        # Output paths using the same base filename
        base_name = os.path.splitext(filename)[0]
        output_image_path = os.path.join(output_body_dir, f"{base_name}_output.jpg")
        
        # Skip if output already exists
        if (os.path.exists(output_image_path)) :
            #print(f"✅ Skipping {filename} (already processed).")
            bodies_already_processed = bodies_already_processed +1
            continue

        try:
            print(f"\n🚀 Processing {filename}...")
            process_body(image_path, output_image_path)
            bodies_newly_processed = bodies_newly_processed +1
            # Optional: display the result visually (can be commented out for batch runs)
            #display_image(header, f"{filename} - header")
            #display_image(body, f"{filename} - body")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print (f"\n{bodies_already_processed} body file(s) left untouched, {bodies_newly_processed} body file(s) processed")


#Process each header file
headers_already_processed = 0
headers_newly_processed =0
for filename in os.listdir(header_dir):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(header_dir, filename)

        # Output paths using the same base filename
        base_name = os.path.splitext(filename)[0]
        output_image_path = os.path.join(output_header_dir, f"{base_name}_output.jpg")
        
        # Skip if output already exists
        if (os.path.exists(output_image_path)) :
            #print(f"✅ Skipping {filename} (already processed).")
            headers_already_processed = headers_already_processed +1
            continue

        try:
            print(f"\n🚀 Processing {filename}...")
            result, boxes = analyze_document_layout(image_path,output_image_path, display_steps= False)
            headers_newly_processed = headers_newly_processed +1
            # Optional: display the result visually (can be commented out for batch runs)
            #display_image(header, f"{filename} - header")
            #display_image(body, f"{filename} - body")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print (f"\n{headers_already_processed} header file(s) left untouched, {headers_newly_processed} header file(s) processed")






#result, boxes = analyze_document_layout('headers/sample1_header.jpg','outputs/test.jpg', display_steps= True)

"""
process_body('bodies/sample1_body.jpg', 'outputs/body_outputs/test1.jpg')
process_body('bodies/sample2_body.jpg', 'outputs/body_outputs/test2.jpg')
process_body('bodies/sample3_body.jpg', 'outputs/body_outputs/test3.jpg')
"""
#visualize_header_detection('images/sample8.jpg')

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
