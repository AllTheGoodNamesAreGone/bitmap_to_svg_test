import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path
import shutil
import pytesseract
from pytesseract import Output
import re

# Page configuration
st.set_page_config(
    page_title="Algorithmic Layout Analysis Demo",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Algorithmic Layout Analysis Pipeline")
st.markdown("Upload a document image to see algorithmic layout detection in action!")

# Create temporary directories
TEMP_DIR = Path("temp_algorithmic")
INPUT_DIR = TEMP_DIR / "input"
OUTPUT_DIR = TEMP_DIR / "output"
HEADERS_DIR = OUTPUT_DIR / "headers"
BODIES_DIR = OUTPUT_DIR / "bodies"
SPLIT_DIR = OUTPUT_DIR / "split_images"

# Setup directories
for dir_path in [TEMP_DIR, INPUT_DIR, OUTPUT_DIR, HEADERS_DIR, BODIES_DIR, SPLIT_DIR]:
    dir_path.mkdir(exist_ok=True)

def detect_header_with_instructions_and_show_boxes(image_path, header_path='headers/test.jpg', body_path='bodies/test.jpg', split_image_path='split_images'):
    """
    Your header/body detection function - imported from your code
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
    instruction_regex_alt = re.compile(r'candidates', re.IGNORECASE)
    
    # Group text by line (using top coordinate and height)
    lines = {}
    detected_texts = []  # Store all detected text for display
    
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:  # Only consider text with decent confidence
            text = data['text'][i]
            if text.strip():  # Skip empty text
                top = data['top'][i]
                height = data['height'][i]
                left = data['left'][i]
                width = data['width'][i]
                conf = data['conf'][i]
                
                # Store for display
                detected_texts.append({
                    'text': text,
                    'bbox': (left, top, width, height),
                    'confidence': conf
                })
                
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
    
    # Search for instruction line in each detected line of text
    instruction_found = False
    for line_id, line_data in lines.items():
        line_text = ' '.join(line_data['texts']).lower()
        if (instruction_regex.search(line_text) or instruction_regex_alt.search(line_text)):
            instruction_line_y = line_data['bottom']
            instruction_found = True
            break
    
    # If we found the instruction line, use it as boundary
    if instruction_line_y:
        header_boundary = instruction_line_y
    else:
        # Simple fallback - use 1/3 of image height
        header_boundary = h // 3
    
    # Create header and body images
    header_img = img[0:header_boundary, :]
    body_img = img[header_boundary:, :]
    
    # Save header and body
    cv2.imwrite(header_path, header_img)
    cv2.imwrite(body_path, body_img)
    
    # Drawing boundary on copy image 
    img_with_line = img.copy()
    cv2.line(img_with_line, (0, header_boundary), (img.shape[1], header_boundary), (0, 0, 255), 3)
    cv2.putText(img_with_line, "Header/Body Boundary", (10, header_boundary - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Save boundary visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    boundary_path = os.path.join(split_image_path, f"{base_name}_boundary.jpg")
    cv2.imwrite(boundary_path, img_with_line)
    
    return {
        'header_img': header_img,
        'body_img': body_img,
        'boundary_img': img_with_line,
        'header_boundary': header_boundary,
        'instruction_found': instruction_found,
        'detected_texts': detected_texts,
        'lines': lines,
        'paths': {
            'header': header_path,
            'body': body_path,
            'boundary': boundary_path
        }
    }

def process_body(image_path, output_path, adaptive_block_size=11, adaptive_c=5, min_width=80, min_height=20):
    """
    Process body region to detect text elements using contour detection
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
    
    # Optional morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # List to store bounding boxes
    boxes = []
    
    # Loop through contours to get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_width and h > min_height:  # Filter based on size
            boxes.append((x, y, w, h))
    
    # Create visualization
    img_with_boxes = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the output image with bounding boxes
    cv2.imwrite(output_path, img_with_boxes)
    
    return {
        'original': img,
        'gray': gray,
        'thresh': thresh,
        'dilated': dilated,
        'boxes': boxes,
        'visualization': img_with_boxes,
        'output_path': output_path
    }

def visualize_ocr_detection(image, detected_texts, show_confidence=True):
    """Create visualization of OCR detected text with bounding boxes"""
    img_viz = image.copy()
    
    for item in detected_texts:
        left, top, width, height = item['bbox']
        conf = item['confidence']
        text = item['text']
        
        # Color based on confidence
        if conf > 70:
            color = (0, 255, 0)  # Green for high confidence
        elif conf > 50:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw bounding box
        cv2.rectangle(img_viz, (left, top), (left + width, top + height), color, 2)
        
        # Add confidence score if requested
        if show_confidence:
            label = f"{conf:.0f}%"
            cv2.putText(img_viz, label, (left, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img_viz

# Sidebar controls
st.sidebar.header("📁 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose a document image", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

st.sidebar.header("🔧 Analysis Options")
show_ocr_boxes = st.sidebar.checkbox("Show OCR Text Detection", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
ocr_confidence_threshold = st.sidebar.slider("OCR Confidence Threshold", 
                                            min_value=0, max_value=100, value=30)

# Body processing parameters
st.sidebar.header("📄 Body Processing Parameters")
run_body_processing = st.sidebar.checkbox("Enable Body Element Detection", value=True)
adaptive_block_size = st.sidebar.slider("Adaptive Threshold Block Size", 
                                       min_value=3, max_value=51, value=11, step=2)
adaptive_c = st.sidebar.slider("Adaptive Threshold C Value", 
                              min_value=1, max_value=20, value=5)
min_box_width = st.sidebar.slider("Minimum Box Width", 
                                 min_value=10, max_value=200, value=80)
min_box_height = st.sidebar.slider("Minimum Box Height", 
                                  min_value=5, max_value=100, value=20)

# Main processing
if uploaded_file is not None:
    # Save uploaded file
    input_path = INPUT_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Display original image
    st.subheader("📄 Original Document")
    original_image = Image.open(input_path)
    st.image(original_image, caption="Input Document", width=700)
    
    # Processing button
    if st.sidebar.button("🚀 Run Algorithmic Analysis", type="primary"):
        
        with st.spinner("Analyzing document layout... This may take a moment."):
            
            try:
                # Set up paths
                header_path = str(HEADERS_DIR / f"header_{uploaded_file.name}")
                body_path = str(BODIES_DIR / f"body_{uploaded_file.name}")
                split_path = str(SPLIT_DIR)
                
                # Run header/body detection
                results = detect_header_with_instructions_and_show_boxes(
                    str(input_path), header_path, body_path, split_path
                )
                
                st.success("✅ Layout analysis completed!")
                
                # Display results
                st.subheader("🎯 Header/Body Separation Results")
                
                # Show detection status
                col1, col2 = st.columns(2)
                with col1:
                    if results['instruction_found']:
                        st.success("🎯 'Instructions to Candidates' line detected!")
                    else:
                        st.warning("⚠️ Instructions line not found - using fallback detection")
                
                with col2:
                    st.info(f"📏 Header boundary at y = {results['header_boundary']} pixels")
                
                # Display boundary visualization
                st.subheader("📊 Boundary Detection Visualization")
                boundary_img_pil = Image.open(results['paths']['boundary'])
                st.image(boundary_img_pil, caption="Header/Body Boundary Detection", width=700)
                
                # Display separated regions
                st.subheader("📑 Separated Regions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Header Region**")
                    header_img_pil = Image.open(results['paths']['header'])
                    st.image(header_img_pil, caption="Detected Header", width=400)
                    
                    # Download button
                    with open(results['paths']['header'], "rb") as file:
                        st.download_button(
                            label="📥 Download Header",
                            data=file.read(),
                            file_name=f"header_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
                
                with col2:
                    st.markdown("**Body Region**")
                    body_img_pil = Image.open(results['paths']['body'])
                    st.image(body_img_pil, caption="Detected Body", width=400)
                    
                    # Download button
                    with open(results['paths']['body'], "rb") as file:
                        st.download_button(
                            label="📥 Download Body",
                            data=file.read(),
                            file_name=f"body_{uploaded_file.name}",
                            mime="image/jpeg"
                        )
                
                # OCR Analysis Section
                if show_ocr_boxes:
                    st.subheader("🔍 OCR Text Detection Analysis")
                    
                    # Filter by confidence threshold
                    filtered_texts = [t for t in results['detected_texts'] 
                                    if t['confidence'] >= ocr_confidence_threshold]
                    
                    st.info(f"📊 Found {len(filtered_texts)} text elements above {ocr_confidence_threshold}% confidence")
                    
                    # Create OCR visualization
                    original_cv = cv2.imread(str(input_path))
                    ocr_viz = visualize_ocr_detection(original_cv, filtered_texts, show_confidence)
                    
                    # Convert to PIL for display
                    ocr_viz_rgb = cv2.cvtColor(ocr_viz, cv2.COLOR_BGR2RGB)
                    ocr_viz_pil = Image.fromarray(ocr_viz_rgb)
                    st.image(ocr_viz_pil, caption="OCR Text Detection (Green=High Conf, Yellow=Med, Red=Low)", width=700)
                    
                    # Show detected lines
                    with st.expander("📋 Detected Text Lines"):
                        for line_id in sorted(results['lines'].keys()):
                            line_text = ' '.join(results['lines'][line_id]['texts'])
                            st.write(f"**Line {line_id}:** {line_text}")
                
                # Body Processing Section
                if run_body_processing:
                    st.subheader("📄 Body Element Detection")
                    
                    with st.spinner("Processing body region for element detection..."):
                        
                        # Set up body processing paths
                        body_output_path = str(OUTPUT_DIR / f"body_processed_{uploaded_file.name}")
                        
                        # Run body processing
                        body_results = process_body(
                            results['paths']['body'],
                            body_output_path,
                            adaptive_block_size=adaptive_block_size,
                            adaptive_c=adaptive_c,
                            min_width=min_box_width,
                            min_height=min_box_height
                        )
                        
                        if body_results:
                            st.success(f"✅ Found {len(body_results['boxes'])} text elements in body region")
                            
                            # Display processing stages
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Adaptive Threshold**")
                                thresh_pil = Image.fromarray(body_results['thresh'])
                                st.image(thresh_pil, caption="Binary Threshold", width=400)
                            
                            with col2:
                                st.markdown("**Detected Elements**")
                                viz_rgb = cv2.cvtColor(body_results['visualization'], cv2.COLOR_BGR2RGB)
                                viz_pil = Image.fromarray(viz_rgb)
                                st.image(viz_pil, caption=f"Found {len(body_results['boxes'])} Elements", width=400)
                            
                            # Element statistics
                            if body_results['boxes']:
                                box_areas = [w * h for x, y, w, h in body_results['boxes']]
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Total Elements", len(body_results['boxes']))
                                with col2:
                                    st.metric("Avg Area", f"{np.mean(box_areas):.0f} px²")
                                with col3:
                                    st.metric("Largest Element", f"{max(box_areas):.0f} px²")
                            
                            # Download processed body
                            with open(body_output_path, "rb") as file:
                                st.download_button(
                                    label="📥 Download Processed Body",
                                    data=file.read(),
                                    file_name=f"body_processed_{uploaded_file.name}",
                                    mime="image/jpeg"
                                )
                            
                            # Show detected boxes details
                            with st.expander("📊 Detected Element Details"):
                                for i, (x, y, w, h) in enumerate(body_results['boxes']):
                                    st.write(f"**Element {i+1}:** Position=({x}, {y}), Size={w}×{h}, Area={w*h} px²")
                        
                        else:
                            st.error("❌ Body processing failed")
                
                # Analysis Summary
                with st.expander("📊 Complete Analysis Summary"):
                    summary_data = {
                        "Total Text Elements (OCR)": len(results['detected_texts']),
                        "High Confidence OCR (>70%)": len([t for t in results['detected_texts'] if t['confidence'] > 70]),
                        "Medium Confidence OCR (50-70%)": len([t for t in results['detected_texts'] if 50 <= t['confidence'] <= 70]),
                        "Low Confidence OCR (<50%)": len([t for t in results['detected_texts'] if t['confidence'] < 50]),
                        "Text Lines Detected": len(results['lines']),
                        "Instructions Line Found": results['instruction_found'],
                        "Header Height (pixels)": results['header_boundary']
                    }
                    
                    if run_body_processing and 'body_results' in locals() and body_results:
                        summary_data.update({
                            "Body Elements (Contour)": len(body_results['boxes']),
                            "Processing Parameters": f"Block={adaptive_block_size}, C={adaptive_c}, MinSize={min_box_width}×{min_box_height}"
                        })
                    
                    for key, value in summary_data.items():
                        st.write(f"**{key}:** {value}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.code(f"Error details: {e}")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a document image to begin algorithmic analysis")
    
    st.subheader("📋 Algorithmic Layout Analysis Pipeline")
    st.markdown("""
    This demo showcases the **algorithmic approach** to document layout analysis:
    
    **Analysis Pipeline:**
    1. **OCR Text Detection** - Extract all text with bounding boxes and confidence scores
    2. **Line Grouping** - Group detected text into logical lines
    3. **Instructions Detection** - Search for "Instructions to Candidates" or "Candidates" text
    4. **Header/Body Separation** - Split document at the instructions line
    5. **Body Element Detection** - Use adaptive thresholding + contour detection to find text blocks
    6. **Visualization** - Show all detected boundaries and elements
    
    **Two-Stage Approach:**
    - **Stage 1: OCR-based** - Uses Tesseract for text detection and line grouping
    - **Stage 2: Contour-based** - Uses image processing to detect text blocks in body region
    
    **Tunable Parameters:**
    - **OCR confidence threshold** for filtering low-quality text detection
    - **Adaptive threshold parameters** for body processing  
    - **Minimum element size filters** to remove noise
    
    **Features:**
    - **Real-time parameter adjustment** with immediate visual feedback
    - **Dual detection methods** - OCR + computer vision approaches
    - **Detailed analysis breakdown** with statistics and metrics
    - **Downloadable results** for all processing stages
    
    **Upload an image above** to see the complete algorithmic pipeline in action!
    """)

# Add import for numpy at the top if not already present
import numpy as np

# Cleanup function
def cleanup_temp_files():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

import atexit
atexit.register(cleanup_temp_files)