#!/usr/bin/env python3
"""
Simple Streamlit Dashboard - No syntax errors version
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
from PIL import Image

# Add src to path
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Question Paper Analysis",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 Question Paper Analysis Dashboard")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page:",
            ["Home", "Upload & Analyze", "View Results", "Settings"]
        )
    
    # Page routing
    if page == "Home":
        show_home()
    elif page == "Upload & Analyze":
        show_upload()
    elif page == "View Results":
        show_results()
    elif page == "Settings":
        show_settings()

def show_home():
    st.header("🏠 Home")
    
    st.markdown("""
    ## Welcome to Question Paper Analysis
    
    This tool analyzes question papers using deep learning models.
    
    ### Pipeline Steps:
    1. **Advanced Analysis** - Layout detection with LayoutParser
    2. **Strategy Comparison** - Multiple detection approaches
    3. **Hybrid Detection** - Maximum detail extraction
    4. **Table Parsing** - Cell-level table analysis
    5. **SVG Creation** - Editable vector output
    6. **Visualizations** - Comprehensive analysis views
    """)
    
    # Check for existing results
    results_dir = Path("pipeline_results")
    if results_dir.exists():
        summary_file = results_dir / "PIPELINE_SUMMARY.txt"
        if summary_file.exists():
            st.success("✅ Previous analysis found!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Analysis Complete", "Yes")
            with col2:
                st.metric("Results Available", "6 Steps")
            with col3:
                st.metric("Status", "Ready")

def show_upload():
    st.header("📤 Upload & Analyze")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a question paper image",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        with col2:
            st.subheader("Analysis Options")
            
            # Step selection
            steps = {
                "Advanced Analysis": st.checkbox("Advanced Analysis", value=True),
                "Strategy Comparison": st.checkbox("Strategy Comparison", value=True),
                "Hybrid Detection": st.checkbox("Hybrid Detection", value=True),
                "Table Parsing": st.checkbox("Table Parsing", value=True),
                "SVG Creation": st.checkbox("SVG Creation", value=True),
                "Visualizations": st.checkbox("Visualizations", value=True)
            }
            
            # Advanced options
            with st.expander("Advanced Options"):
                confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
                use_gpu = st.checkbox("Use GPU", value=False)
            
            # Run analysis
            if st.button("🚀 Start Analysis", type="primary"):
                run_analysis_simple(uploaded_file, steps, confidence, use_gpu)

def run_analysis_simple(uploaded_file, steps, confidence, use_gpu):
    """Simple analysis runner with progress tracking"""
    
    # Save uploaded file
    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    image_path = upload_dir / uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Progress tracking
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Import pipeline - adjust import name to match your file
        from master import MasterAnalysisPipeline
        
        # Initialize
        status.text("Initializing pipeline...")
        pipeline = MasterAnalysisPipeline(str(image_path), "pipeline_results")
        
        # Run selected steps
        step_count = 0
        total_steps = sum(steps.values())
        results = {}
        
        if steps["Advanced Analysis"]:
            step_count += 1
            progress.progress(step_count / total_steps)
            status.text("Running Advanced Analysis...")
            
            result = pipeline.run_advanced_analysis()
            results['advanced'] = result
            
            if result['success']:
                st.success(f"✅ Advanced Analysis: {result['blocks_count']} blocks")
            else:
                st.error(f"❌ Advanced Analysis failed: {result['error']}")
        
        if steps["Strategy Comparison"]:
            step_count += 1
            progress.progress(step_count / total_steps)
            status.text("Running Strategy Comparison...")
            
            result = pipeline.run_strategy_comparison()
            results['strategy'] = result
            
            if result['success']:
                st.success(f"✅ Strategy Comparison: {result['best_strategy']}")
            else:
                st.error(f"❌ Strategy Comparison failed")
        
        if steps["Hybrid Detection"]:
            step_count += 1
            progress.progress(step_count / total_steps)
            status.text("Running Hybrid Detection...")
            
            result = pipeline.run_hybrid_svg_detection()
            results['hybrid'] = result
            
            if result['success']:
                st.success(f"✅ Hybrid Detection: {result['elements_count']} elements")
            else:
                st.error(f"❌ Hybrid Detection failed")
        
        if steps["Table Parsing"]:
            step_count += 1
            progress.progress(step_count / total_steps)
            status.text("Running Table Parsing...")
            
            result = pipeline.run_table_parsing()
            results['table'] = result
            
            if result['success']:
                st.success(f"✅ Table Parsing: {result['cells_count']} cells")
            else:
                st.error(f"❌ Table Parsing failed")
        
        if steps["SVG Creation"]:
            step_count += 1
            progress.progress(step_count / total_steps)
            status.text("Creating SVG...")
            
            result = pipeline.run_svg_creation(results.get('hybrid'))
            results['svg'] = result
            
            if result['success']:
                st.success(f"✅ SVG Creation: Complete")
            else:
                st.error(f"❌ SVG Creation failed")
        
        if steps["Visualizations"]:
            step_count += 1
            progress.progress(step_count / total_steps)
            status.text("Creating Visualizations...")
            
            result = pipeline.run_visualization_enhancement(results.get('advanced'))
            results['viz'] = result
            
            if result['success']:
                st.success(f"✅ Visualizations: {result['visualizations_count']} created")
            else:
                st.error(f"❌ Visualizations failed")
        
        # Complete
        progress.progress(1.0)
        status.text("✅ Analysis Complete!")
        
        # Generate summary
        pipeline.generate_pipeline_summary(results, 60)
        
        st.balloons()
        st.success("🎉 Analysis completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")

def show_results():
    st.header("📊 Analysis Results")
    
    results_dir = Path("pipeline_results")
    
    if not results_dir.exists():
        st.info("No results found. Run an analysis first!")
        return
    
    # Load summary
    summary_file = results_dir / "PIPELINE_SUMMARY.txt"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()
        
        with st.expander("📋 Pipeline Summary"):
            st.text(summary)
    
    # Results tabs
    tabs = st.tabs(["📋 Analysis", "🔄 Strategy", "🎯 Hybrid", "📊 Tables", "🎨 SVG", "📈 Viz"])
    
    with tabs[0]:
        show_analysis_results(results_dir)
    
    with tabs[1]:
        show_strategy_results(results_dir)
    
    with tabs[2]:
        show_hybrid_results(results_dir)
    
    with tabs[3]:
        show_table_results(results_dir)
    
    with tabs[4]:
        show_svg_results(results_dir)
    
    with tabs[5]:
        show_viz_results(results_dir)

def show_analysis_results(results_dir):
    analysis_dir = results_dir / "01_advanced_analysis"
    if not analysis_dir.exists():
        st.info("Advanced analysis results not found.")
        return
    
    # Find files
    json_files = list(analysis_dir.glob("*.json"))
    
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Layout Blocks", len(data.get('layout_blocks', [])))
        with col2:
            st.metric("Tables", len(data.get('tables', [])))
        with col3:
            st.metric("Text Blocks", len(data.get('text_content', {})))
        
        # Block types chart
        if 'layout_blocks' in data:
            blocks = data['layout_blocks']
            types = [block['type'] for block in blocks]
            type_counts = pd.Series(types).value_counts()
            
            fig = px.pie(values=type_counts.values, names=type_counts.index, title="Block Types")
            st.plotly_chart(fig, use_container_width=True)

def show_strategy_results(results_dir):
    strategy_dir = results_dir / "02_strategy_comparison"
    if not strategy_dir.exists():
        st.info("Strategy results not found.")
        return
    
    # Find images
    images = list(strategy_dir.glob("*.png"))
    for img in images:
        st.image(str(img), caption=img.name, use_column_width=True)

def show_hybrid_results(results_dir):
    hybrid_dir = results_dir / "03_hybrid_svg_detection"
    if not hybrid_dir.exists():
        st.info("Hybrid results not found.")
        return
    
    # Find images
    images = list(hybrid_dir.glob("*.png"))
    for img in images:
        st.image(str(img), caption=img.name, use_column_width=True)

def show_table_results(results_dir):
    table_dir = results_dir / "04_table_parsing"
    if not table_dir.exists():
        st.info("Table results not found.")
        return
    
    # Find images
    images = list(table_dir.glob("*.png"))
    for img in images:
        st.image(str(img), caption=img.name, use_column_width=True)

def show_svg_results(results_dir):
    svg_dir = results_dir / "05_svg_creation"
    if not svg_dir.exists():
        st.info("SVG results not found.")
        return
    
    # Find SVG files
    svg_files = list(svg_dir.glob("*.svg"))
    html_files = list(svg_dir.glob("*.html"))
    
    if svg_files:
        st.subheader("🎨 SVG File")
        svg_file = svg_files[0]
        
        with open(svg_file, 'r') as f:
            svg_content = f.read()
        
        st.download_button(
            "📄 Download SVG",
            data=svg_content,
            file_name=svg_file.name,
            mime="image/svg+xml"
        )
    
    if html_files:
        st.subheader("🌐 HTML Preview")
        html_file = html_files[0]
        
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.download_button(
            "🌐 Download HTML",
            data=html_content,
            file_name=html_file.name,
            mime="text/html"
        )

def show_viz_results(results_dir):
    viz_dir = results_dir / "06_visualization_enhancement"
    if not viz_dir.exists():
        st.info("Visualization results not found.")
        return
    
    # Find images
    images = list(viz_dir.glob("*.png"))
    for img in images:
        st.image(str(img), caption=img.name, use_column_width=True)
    
    # Find blocks directory
    blocks_dirs = list(viz_dir.glob("*_blocks"))
    if blocks_dirs:
        blocks_dir = blocks_dirs[0]
        block_files = list(blocks_dir.glob("*.png"))
        
        st.subheader(f"🧩 Individual Blocks ({len(block_files)})")
        
        # Display in grid
        cols = st.columns(4)
        for i, block_file in enumerate(block_files[:12]):  # Show first 12
            with cols[i % 4]:
                st.image(str(block_file), caption=block_file.stem, use_column_width=True)

def show_settings():
    st.header("🔧 Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Settings")
        model = st.selectbox("Layout Model", [
            "publaynet_mask_rcnn_R_50_FPN_3x",
            "publaynet_faster_rcnn_R_50_FPN_3x"
        ])
        
        threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5)
        use_gpu = st.checkbox("Use GPU", value=False)
    
    with col2:
        st.subheader("Output Settings")
        save_blocks = st.checkbox("Save Individual Blocks", value=True)
        high_dpi = st.checkbox("High DPI Output", value=True)
        create_html = st.checkbox("Create HTML Preview", value=True)
    
    if st.button("💾 Save Settings"):
        settings = {
            'model': model,
            'threshold': threshold,
            'use_gpu': use_gpu,
            'save_blocks': save_blocks,
            'high_dpi': high_dpi,
            'create_html': create_html
        }
        
        # Save settings
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        
        with open(settings_dir / "config.json", 'w') as f:
            json.dump(settings, f, indent=2)
        
        st.success("✅ Settings saved!")

if __name__ == "__main__":
    main()