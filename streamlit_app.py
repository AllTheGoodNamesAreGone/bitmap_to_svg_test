import streamlit as st
import os
import shutil
from PIL import Image
import subprocess
import sys
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Document Preprocessing Demo",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Image Preprocessing Pipeline")
st.markdown("Upload a document image to see the preprocessing stages in action!")

# Sidebar for controls and parameters
st.sidebar.header("📁 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

st.sidebar.header("⚙️ Processing Parameters")

# Parameters from your script
post_warp_deskew = st.sidebar.checkbox("Apply Post-Warp Deskew", value=True)

median_kernel = st.sidebar.slider(
    "Median Blur Kernel Size", 
    min_value=1, max_value=15, value=1, step=2,
    help="Must be odd. 1 disables median blur."
)

block_size = st.sidebar.slider(
    "Adaptive Threshold Block Size", 
    min_value=3, max_value=51, value=25, step=2,
    help="Must be odd. Controls local neighborhood size."
)

c_constant = st.sidebar.slider(
    "Adaptive Threshold Constant", 
    min_value=1, max_value=20, value=10,
    help="Constant subtracted from weighted mean."
)

# Package parameters
processing_params = {
    'post_warp_deskew': post_warp_deskew,
    'median_kernel': median_kernel,
    'block_size': block_size,
    'c_constant': c_constant
}

# Create temporary directories
TEMP_DIR = Path("temp_processing")
INPUT_DIR = TEMP_DIR / "input"
OUTPUT_DIR = TEMP_DIR / "output"

def setup_directories():
    """Create necessary directories"""
    TEMP_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

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

# Main processing logic
if uploaded_file is not None:
    setup_directories()
    
    # Save uploaded file
    input_path = INPUT_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    # Display original image
    st.subheader("📷 Original Image")
    original_image = Image.open(input_path)
    st.image(original_image, caption="Input Document", use_container_width=True)
    
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
                    st.image(final_image, caption="Final Preprocessed Output", use_container_width=True)
                    
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
                                st.image(
                                    stage_image, 
                                    caption=stage_name,
                                    use_container_width=True
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
                        st.image(original_image, caption="Original Input", use_container_width=True)
                    
                    with col2:
                        if output_path.exists():
                            final_image = Image.open(output_path)
                            st.image(final_image, caption="Final Output", use_container_width=True)
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
    
    # You can add sample images here
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