import cv2
import os

from get_header_line_based_improved import detect_header_with_instructions_and_show_boxes, visualize_header_detection
from display_image import display_image
from process_image import process_image, process_body
from layout_analyzer_v3 import analyze_document_layout, detect_logo, detect_tables, detect_text_improved, merge_close_text_boxes, merge_overlapping_boxes
import combined, combined2
import headertrial
import template_based_detector
import highlight_regions

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

print (f" \n ➡️\t {already_processed} image file(s) left untouched, {newly_processed} image file(s) processed")


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

print (f"\n ➡️\t {bodies_already_processed} body file(s) left untouched, {bodies_newly_processed} body file(s) processed")

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

print (f"\n ➡️\t {headers_already_processed} header file(s) left untouched, {headers_newly_processed} header file(s) processed\n")


#Template based attempt 
#detector = template_based_detector.TemplateBasedDetector('templates/ramaiah_template.json')
#detector.load_image("images/sample7.jpg")
#results = detector.detect_all_regions()
#detector.generate_svg_with_detections('output.svg', results)



#Highlighting only 
#highlight_regions.generate_highlighted_question_paper("images/sample1.jpg", "highlighted_output11.jpg")

import create_svg_2

#create_svg_2.generate_content_svg("images/sample5.jpg", "output6.svg", debug_tables=True)


import table_detection

#table_detection.debug_table_detection_with_svg("images/sample6.jpg", "tables_only6.svg", "debug_tables_only6.jpg")


import table_detection_2

table_detection_2.debug_table_detection_with_svg("images/sample1.jpg", "outputs/tables/tables_only1v2.svg", "outputs/tables/debug/debug_tables_only1v2.jpg")
#table_detection.debug_table_detection_with_svg("images/sample11.jpg", "outputs/tables/tables_only11.svg", "outputs/tables/debug/debug_tables_only11.jpg")
#table_detection.debug_table_detection_with_svg("images/sample12.jpg", "outputs/tables/tables_only12.svg", "outputs/tables/debug/debug_tables_only12.jpg")
#table_detection.debug_table_detection_with_svg("images/sample13.jpg", "outputs/tables/tables_only13.svg", "outputs/tables/debug/debug_tables_only13.jpg")
#Miscellaneous testing below this ------------------------------------------------------------------------------------------------------------------------------------

#combined.generate_question_paper_svg("images\sample1.jpg", "outputs/testing/combined1.svg")
#combined.generate_question_paper_svg("images\sample2.jpg", "outputs/testing/combined2.svg")
#combined2.process_exam_paper("images\sample1.jpg", "outputs/testing/combined22.svg")
#combined.generate_question_paper_svg("images\sample6.jpg", "outputs/testing/combined6.svg")
#combined.generate_question_paper_svg("images\sample7.jpg", "outputs/testing/combined7.svg")
#combined.generate_question_paper_svg("images\sample8.jpg", "outputs/testing/combined8.svg")

#result, boxes = analyze_document_layout("headers/sample1_header.jpg","outputs/header1repeat.jpg", display_steps= False)

#headertrial.generate_svg_from_detected_elements("headers/sample1_header.jpg", boxes, output_svg_path="outputs/header1trial.svg")

#process_body("bodies/sample15_body.jpg", "outputs/test.jpg")

#combined2.process_exam_paper("D:\Anirudh\mini_project_2\images\sample1.jpg","outputs/testing/sample1_output_full.svg")

#result, boxes = v2.analyze_document_layout('headers/sample14_header.jpg','outputs/test.jpg', display_steps= True)


