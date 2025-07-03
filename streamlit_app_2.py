import streamlit as st
import os
import shutil
from PIL import Image
import subprocess
import sys
import cv2
import numpy as np
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Document Preprocessing Demo",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Image Preprocessing Pipeline")
st.markdown("Upload a document image to see the preprocessing stages in action!")

# Create temporary directories at startup
TEMP_DIR = Path("temp_processing")
INPUT_DIR = TEMP_DIR / "input"
OUTPUT_DIR = TEMP_DIR / "output"

# Setup directories immediately
TEMP_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_image_quality(image):
    """Analyze image quality metrics"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Contrast (standard deviation)
    contrast = gray.std()
    
    # Brightness (mean)
    brightness = gray.mean()
    
    # Noise estimation (difference between original and median filtered)
    denoised = cv2.medianBlur(gray, 5)
    noise = np.mean((gray.astype(float) - denoised.astype(float))**2)
    
    return {
        'sharpness': round(sharpness, 2),
        'contrast': round(contrast, 2), 
        'brightness': round(brightness, 2),
        'noise': round(noise, 2)
    }

def suggest_parameters(image):
    """Auto-suggest parameters based on image analysis"""
    metrics = analyze_image_quality(image)
    
    suggestions = {
        'post_warp_deskew': True,  # Generally good to have
        'median_kernel': 1,  # Default
        'block_size': 25,    # Default
        'c_constant': 10     # Default
    }
    
    # Adjust based on noise level
    if metrics['noise'] > 50:
        suggestions['median_kernel'] = 5  # More aggressive denoising
    elif metrics['noise'] > 20:
        suggestions['median_kernel'] = 3
    
    # Adjust based on contrast
    if metrics['contrast'] < 30:
        suggestions['c_constant'] = 5  # Lower constant for low contrast
        suggestions['block_size'] = 15  # Smaller block size
    elif metrics['contrast'] > 80:
        suggestions['c_constant'] = 15  # Higher constant for high contrast
    
    # Adjust based on brightness
    if metrics['brightness'] < 100:  # Dark image
        suggestions['c_constant'] = max(3, suggestions['c_constant'] - 3)
    elif metrics['brightness'] > 180:  # Bright image
        suggestions['c_constant'] = min(20, suggestions['c_constant'] + 3)
    
    return suggestions, metrics

def create_overlay_image(original_img, lines_data=None, corners_data=None):
    """Create overlay visualization with detected lines and corners"""
    overlay = original_img.copy()
    
    # This is a placeholder - you'd need to modify your preprocessing script
    # to return the detected lines and corners data
    if lines_data:
        # Draw horizontal lines in red
        for line in lines_data.get('horizontal', []):
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw vertical lines in blue  
        for line in lines_data.get('vertical', []):
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    if corners_data:
        # Draw corners as circles
        for corner in corners_data:
            cv2.circle(overlay, tuple(corner), 10, (0, 255, 0), -1)
    
    return overlay

def run_preprocessing(input_path, output_path, debug_folder, params):
    """
    Run your preprocessing script by importing and calling the function directly
    """
    try:
        # Import your preprocessing function
        # Make sure your script file is in the same directory or adjust the import path
        from your_preprocessing_script import preprocess_document_hough_v15
        
        # Call the preprocessing function with parameters
        success = preprocess_document_hough_v15(
            input_path=str(input_path),
            output_path=str(output_path),
            debug_folder=debug_folder,
            apply_post_warp_deskew=params['post_warp_deskew'],
            median_blur_kernel=params['median_kernel'],
            adaptive_thresh_blocksize=params['block_size'],
            adaptive_thresh_C=params['c_constant']
        )
        
        return success, "Processing completed", ""
        
    except Exception as e:
        return False, "", str(e)

def display_processing_stages(debug_folder):
    """Display all intermediate processing stages based on your script's debug output"""
    
    # Based on your script's save_debug_image calls
    stage_files = {
        "01 - Original Image": "01_original.png",
        "02 - Edge Detection": "02_edged.png", 
        "07 - Warped Color": "07_warped_color.png",
        "08 - Warped Grayscale": "08_warped_gray.png",
        "08a - Post-Warp Deskewed": "08a_post_warp_deskewed.png",
        "08b - Median Blurred": "08b_median_blurred.png",
        "09 - Final Binary": "09_final_binary.png"
    }
    
    stages_found = {}
    debug_path = Path(debug_folder)
    
    for stage_name, filename in stage_files.items():
        file_path = debug_path / filename
        if file_path.exists():
            stages_found[stage_name] = file_path
    
    return stages_found

# Sidebar for controls and parameters
st.sidebar.header("📁 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

# Auto-suggest parameters when image is uploaded
if uploaded_file is not None and 'suggestions' not in st.session_state:
    # Save uploaded file temporarily for analysis
    temp_path = INPUT_DIR / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load image for analysis
    img_for_analysis = cv2.imread(str(temp_path))
    if img_for_analysis is not None:
        suggestions, quality_metrics = suggest_parameters(img_for_analysis)
        st.session_state.suggestions = suggestions
        st.session_state.quality_metrics = quality_metrics

st.sidebar.header("⚙️ Processing Parameters")

# Show auto-suggestions if available
if 'suggestions' in st.session_state:
    st.sidebar.success("🤖 Auto-suggestions applied!")
    if st.sidebar.button("📊 View Image Analysis"):
        st.session_state.show_analysis = True

# Parameters from your script with auto-suggestions
default_deskew = st.session_state.get('suggestions', {}).get('post_warp_deskew', True)
default_median = st.session_state.get('suggestions', {}).get('median_kernel', 1)
default_block = st.session_state.get('suggestions', {}).get('block_size', 25)
default_constant = st.session_state.get('suggestions', {}).get('c_constant', 10)

post_warp_deskew = st.sidebar.checkbox("Apply Post-Warp Deskew", value=default_deskew)

median_kernel = st.sidebar.slider(
    "Median Blur Kernel Size", 
    min_value=1, max_value=15, value=default_median, step=2,
    help="Must be odd. 1 disables median blur."
)

block_size = st.sidebar.slider(
    "Adaptive Threshold Block Size", 
    min_value=3, max_value=51, value=default_block, step=2,
    help="Must be odd. Controls local neighborhood size."
)

c_constant = st.sidebar.slider(
    "Adaptive Threshold Constant", 
    min_value=1, max_value=20, value=default_constant,
    help="Constant subtracted from weighted mean."
)

# Display options
st.sidebar.header("🔍 Display Options")
show_overlays = st.sidebar.checkbox("Show Line/Corner Overlays", value=False)
zoom_level = st.sidebar.slider("Zoom Level", min_value=50, max_value=200, value=100, step=10)

# Package parameters
processing_params = {
    'post_warp_deskew': post_warp_deskew,
    'median_kernel': median_kernel,
    'block_size': block_size,
    'c_constant': c_constant
}

# Main processing logic
if uploaded_file is not None:
    # Save uploaded file
    input_path = INPUT_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Display image analysis if requested
    if st.session_state.get('show_analysis', False):
        st.subheader("📊 Image Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics = st.session_state.quality_metrics
            st.metric("Sharpness", f"{metrics['sharpness']:.1f}", help="Higher = sharper image")
            st.metric("Contrast", f"{metrics['contrast']:.1f}", help="Higher = more contrast")
        
        with col2:
            st.metric("Brightness", f"{metrics['brightness']:.1f}", help="0-255 scale")
            st.metric("Noise Level", f"{metrics['noise']:.1f}", help="Lower = less noise")
        
        # Recommendations
        st.info("🤖 **Auto-suggestions applied based on analysis:**")
        suggestions = st.session_state.suggestions
        st.write(f"• Median Blur: {suggestions['median_kernel']} (based on noise level)")
        st.write(f"• Block Size: {suggestions['block_size']} (based on contrast)")
        st.write(f"• Threshold Constant: {suggestions['c_constant']} (based on brightness)")
    
    # Display original image with zoom and overlays
    st.subheader("📷 Original Image")
    
    original_image = Image.open(input_path)
    
    # Apply zoom
    if zoom_level != 100:
        width, height = original_image.size
        new_width = int(width * zoom_level / 100)
        new_height = int(height * zoom_level / 100)
        display_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        display_image = original_image
    
    # Create overlay if requested (placeholder functionality)
    if show_overlays:
        st.info("🔍 Overlay mode: Line and corner detection overlays will be shown after processing")
    
    # Use smaller width for better screen fit
    st.image(display_image, caption="Input Document", width=600)
    
    # Processing button
    if st.sidebar.button("🚀 Run Preprocessing", type="primary"):
        
        with st.spinner("Processing image... This may take a few moments."):
            
            # Set up output paths
            output_path = OUTPUT_DIR / f"processed_{uploaded_file.name}"
            debug_folder = str(OUTPUT_DIR / "debug_stages")
            
            # Run preprocessing
            success, stdout, stderr = run_preprocessing(
                input_path, output_path, debug_folder, processing_params
            )
            
            if success:
                st.success("✅ Preprocessing completed successfully!")
                
                # Display final output first
                if output_path.exists():
                    st.subheader("🎯 Final Processed Image")
                    final_image = Image.open(output_path)
                    
                    # Apply zoom to final image
                    if zoom_level != 100:
                        width, height = final_image.size
                        new_width = int(width * zoom_level / 100)
                        new_height = int(height * zoom_level / 100)
                        final_display = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    else:
                        final_display = final_image
                    
                    st.image(final_display, caption="Final Preprocessed Output", width=600)
                    
                    # Quality metrics comparison
                    col1, col2, col3 = st.columns(3)
                    
                    # Calculate metrics for processed image
                    final_img_cv = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
                    if final_img_cv is not None:
                        final_metrics = analyze_image_quality(final_img_cv)
                        original_metrics = st.session_state.quality_metrics
                        
                        with col1:
                            st.metric(
                                "Sharpness", 
                                f"{final_metrics['sharpness']:.1f}",
                                delta=f"{final_metrics['sharpness'] - original_metrics['sharpness']:.1f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Contrast", 
                                f"{final_metrics['contrast']:.1f}",
                                delta=f"{final_metrics['contrast'] - original_metrics['contrast']:.1f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Noise Level", 
                                f"{final_metrics['noise']:.1f}",
                                delta=f"{final_metrics['noise'] - original_metrics['noise']:.1f}",
                                delta_color="inverse"
                            )
                    
                    # Download button for final output
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Final Output",
                            data=file.read(),
                            file_name=f"processed_{uploaded_file.name}",
                            mime="image/png"
                        )
                
                # Display processing stages
                stages = display_processing_stages(debug_folder)
                
                if stages:
                    st.subheader("🔄 Processing Pipeline Stages")
                    
                    # Create expandable sections for better organization
                    for stage_name, img_path in stages.items():
                        with st.expander(f"View: {stage_name}", expanded=False):
                            try:
                                stage_image = Image.open(img_path)
                                
                                # Apply zoom to stage images
                                if zoom_level != 100:
                                    width, height = stage_image.size
                                    new_width = int(width * zoom_level / 100)
                                    new_height = int(height * zoom_level / 100)
                                    stage_display = stage_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                else:
                                    stage_display = stage_image
                                
                                st.image(
                                    stage_display, 
                                    caption=stage_name,
                                    width=600
                                )
                                
                                # Download button for each stage
                                with open(img_path, "rb") as file:
                                    st.download_button(
                                        label=f"Download {stage_name}",
                                        data=file.read(),
                                        file_name=f"{stage_name.lower().replace(' ', '_').replace('-', '_')}.png",
                                        mime="image/png",
                                        key=f"download_{stage_name}"
                                    )
                            except Exception as e:
                                st.error(f"Error loading {stage_name}: {str(e)}")
                    
                    # Before/After Comparison
                    st.subheader("📊 Before vs After Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Apply zoom to comparison images
                        if zoom_level != 100:
                            width, height = original_image.size
                            new_width = int(width * zoom_level / 100)
                            new_height = int(height * zoom_level / 100)
                            orig_display = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        else:
                            orig_display = original_image
                        st.image(orig_display, caption="Original Input", width=400)
                    
                    with col2:
                        if output_path.exists():
                            if zoom_level != 100:
                                final_comp = final_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            else:
                                final_comp = final_image
                            st.image(final_comp, caption="Final Output", width=400)
                        else:
                            st.error("Final output not found")
                
                # Show processing parameters used
                with st.expander("⚙️ Processing Parameters Used"):
                    st.json(processing_params)
                        
            else:
                st.error("❌ Preprocessing failed!")
                if stderr:
                    st.error(stderr)
                
                # Show debug info
                with st.expander("🐛 Debug Information"):
                    st.code(f"Error: {stderr}")
                    if stdout:
                        st.code(f"Output: {stdout}")

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an image file to begin preprocessing")
    
    st.subheader("📋 Document Image Preprocessing Pipeline")
    st.markdown("""
    This demo showcases the **Hough-based perspective correction** preprocessing pipeline:
    
    **Pipeline Steps:**
    1. **Edge Detection** - Canny edge detection to find document boundaries
    2. **Line Detection** - Hough transform to detect horizontal/vertical lines  
    3. **Corner Finding** - Intersection of detected lines to find document corners
    4. **Perspective Correction** - Warp document to rectangular shape
    5. **Optional Deskewing** - Remove rotational skew after warping
    6. **Median Filtering** - Remove noise and scratches (optional)
    7. **Adaptive Binarization** - Convert to clean binary image
    
    **Tunable Parameters:**
    - **Post-Warp Deskew**: Apply rotational correction after perspective transform
    - **Median Blur**: Remove small noise/scratches (kernel size, odd numbers only)
    - **Adaptive Threshold**: Fine-tune binarization (block size and constant)
    
    **Upload an image above** to see each step in action!
    """)

# Cleanup on app restart
def cleanup_temp_files():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

# Register cleanup
import atexit
atexit.register(cleanup_temp_files)