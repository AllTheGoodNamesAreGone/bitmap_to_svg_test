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
                
                # Analysis Summary
                with st.expander("📊 Analysis Summary"):
                    summary_data = {
                        "Total Text Elements": len(results['detected_texts']),
                        "High Confidence (>70%)": len([t for t in results['detected_texts'] if t['confidence'] > 70]),
                        "Medium Confidence (50-70%)": len([t for t in results['detected_texts'] if 50 <= t['confidence'] <= 70]),
                        "Low Confidence (<50%)": len([t for t in results['detected_texts'] if t['confidence'] < 50]),
                        "Text Lines Detected": len(results['lines']),
                        "Instructions Line Found": results['instruction_found'],
                        "Header Height (pixels)": results['header_boundary']
                    }
                    
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
    
    **Analysis Steps:**
    1. **OCR Text Detection** - Extract all text with bounding boxes and confidence scores
    2. **Line Grouping** - Group detected text into logical lines
    3. **Instructions Detection** - Search for "Instructions to Candidates" or "Candidates" text
    4. **Header/Body Separation** - Split document at the instructions line
    5. **Fallback Method** - Use simple heuristics if instructions line not found
    6. **Visualization** - Show detected boundaries and text regions
    
    **Features:**
    - **Real-time OCR analysis** with confidence scoring
    - **Visual boundary detection** with color-coded confidence levels
    - **Automatic header/body separation** 
    - **Detailed text line analysis**
    - **Downloadable separated regions**
    
    **Upload an image above** to see the algorithmic approach in action!
    """)

# Cleanup function
def cleanup_temp_files():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

import atexit
atexit.register(cleanup_temp_files)