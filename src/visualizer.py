#!/usr/bin/env python3
"""
Fixed MockLayoutBlock that's compatible with LayoutParser's drawing functions
"""

import layoutparser as lp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path

class FixedMockLayoutBlock:
    """Mock layout block that's fully compatible with LayoutParser"""
    
    def __init__(self, block_info):
        self.type = block_info['type']
        self.score = block_info['confidence']
        coords = block_info['coordinates']
        self.coordinates = [coords['x1'], coords['y1'], coords['x2'], coords['y2']]
        
        # Add missing attributes that LayoutParser expects
        self.id = block_info.get('id', 0)
        self.block = self  # Some LayoutParser functions expect this
        
        # Additional attributes for full compatibility
        self.x_1 = coords['x1']
        self.y_1 = coords['y1'] 
        self.x_2 = coords['x2']
        self.y_2 = coords['y2']
        self.width = coords['x2'] - coords['x1']
        self.height = coords['y2'] - coords['y1']

class FixedVisualizationEnhancer:
    def __init__(self):
        """Initialize visualization tools with error handling"""
        self.colors = {
            'Text': '#2E86AB',      # Blue
            'Title': '#A23B72',     # Magenta
            'Table': '#F18F01',     # Orange
            'Figure': '#C73E1D',    # Red
            'List': '#6A994E'       # Green
        }
    
    def create_full_visualization(self, image_path, layout_blocks, output_dir):
        """Create comprehensive visualization with proper error handling"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = Path(image_path).stem
        
        print(f"🎨 Creating visualizations for {base_name}...")
        
        # Method 1: Try LayoutParser's built-in drawing (with fixed blocks)
        success_lp = self.try_layoutparser_drawing_safe(image_rgb, layout_blocks, output_dir, base_name)
        
        # Method 2: Always create matplotlib visualization (more reliable)
        success_mpl = self.create_matplotlib_visualization(image_rgb, layout_blocks, output_dir, base_name)
        
        # Method 3: Save individual blocks
        success_blocks = self.save_individual_blocks(image_rgb, layout_blocks, output_dir, base_name)
        
        # Method 4: Create detailed analysis dashboard
        success_dashboard = self.create_analysis_dashboard(image_rgb, layout_blocks, output_dir, base_name)
        
        # Summary
        successes = sum([success_lp, success_mpl, success_blocks, success_dashboard])
        print(f"✅ Created {successes}/4 visualization types successfully")
    
    def try_layoutparser_drawing_safe(self, image, layout_blocks, output_dir, base_name):
        """Safely try LayoutParser's drawing with error handling"""
        try:
            print("  📊 Trying LayoutParser built-in visualization...")
            
            # Create a simple layout collection for LayoutParser
            layout_collection = lp.Layout(layout_blocks)
            
            # Try the drawing
            annotated_image = lp.draw_box(
                image.copy(), 
                layout_collection, 
                box_width=3, 
                show_element_type=True,
                show_element_id=False  # Disable ID to avoid issues
            )
            
            plt.figure(figsize=(16, 20))
            plt.imshow(annotated_image)
            plt.axis('off')
            plt.title(f"LayoutParser Analysis: {base_name}", fontsize=16, pad=20)
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"{base_name}_layoutparser_viz.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ LayoutParser visualization saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"    ⚠️  LayoutParser drawing failed: {e}")
            print("    🔧 Falling back to manual visualization...")
            return False
    
    def create_matplotlib_visualization(self, image, layout_blocks, output_dir, base_name):
        """Create reliable matplotlib visualization"""
        try:
            print("  📊 Creating matplotlib visualization...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
            
            # Original image
            ax1.imshow(image)
            ax1.set_title("Original Image", fontsize=14)
            ax1.axis('off')
            
            # Annotated image
            ax2.imshow(image)
            
            # Draw bounding boxes
            for i, block in enumerate(layout_blocks):
                if hasattr(block, 'coordinates'):
                    # Block is from JSON
                    x1, y1, x2, y2 = block.coordinates
                    block_type = block.type
                    confidence = getattr(block, 'score', 0)
                else:
                    # Block is a dict
                    coords = block['coordinates']
                    x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                    block_type = block['type']
                    confidence = block.get('confidence', 0)
                
                width = x2 - x1
                height = y2 - y1
                
                # Get color for block type
                color = self.colors.get(block_type, '#666666')
                
                # Draw rectangle
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=3, edgecolor=color, facecolor='none',
                    alpha=0.8
                )
                ax2.add_patch(rect)
                
                # Add label with background
                label = f"{i}: {block_type}"
                ax2.text(
                    x1, y1-10, label,
                    fontsize=10, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9)
                )
                
                # Add confidence score
                if confidence > 0:
                    conf_text = f"{confidence:.2f}"
                    ax2.text(
                        x2-40, y1-10, conf_text,
                        fontsize=8, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7)
                    )
            
            ax2.set_title(f"Layout Analysis: {len(layout_blocks)} blocks detected", fontsize=14)
            ax2.axis('off')
            
            # Add legend
            legend_elements = []
            detected_types = set()
            
            for block in layout_blocks:
                if hasattr(block, 'type'):
                    detected_types.add(block.type)
                else:
                    detected_types.add(block['type'])
            
            for block_type in detected_types:
                color = self.colors.get(block_type, '#666666')
                legend_elements.append(
                    patches.Patch(color=color, label=block_type)
                )
            
            if legend_elements:
                ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"{base_name}_matplotlib_viz.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Matplotlib visualization saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"    ❌ Matplotlib visualization failed: {e}")
            return False
    
    def save_individual_blocks(self, image, layout_blocks, output_dir, base_name):
        """Save individual block images with error handling"""
        try:
            print("  📊 Saving individual blocks...")
            
            blocks_dir = os.path.join(output_dir, f"{base_name}_blocks")
            os.makedirs(blocks_dir, exist_ok=True)
            
            saved_blocks = 0
            
            for i, block in enumerate(layout_blocks):
                try:
                    # Handle both object and dict formats
                    if hasattr(block, 'coordinates'):
                        x1, y1, x2, y2 = map(int, block.coordinates)
                        block_type = block.type
                        confidence = getattr(block, 'score', 0)
                    else:
                        coords = block['coordinates']
                        x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                        block_type = block['type']
                        confidence = block.get('confidence', 0)
                    
                    # Validate coordinates
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= image.shape[1] and y2 <= image.shape[0]:
                        # Extract block region
                        block_image = image[y1:y2, x1:x2]
                        
                        if block_image.size > 0:
                            # Create figure for this block
                            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                            ax.imshow(block_image)
                            ax.axis('off')
                            
                            # Title with block info
                            title = f"Block {i}: {block_type}"
                            if confidence > 0:
                                title += f"\nConfidence: {confidence:.3f}"
                            title += f"\nSize: {x2-x1}x{y2-y1}"
                            ax.set_title(title, fontsize=12, pad=10)
                            
                            # Save block image
                            block_path = os.path.join(blocks_dir, f"block_{i:02d}_{block_type.lower()}.png")
                            plt.savefig(block_path, dpi=200, bbox_inches='tight')
                            plt.close()
                            
                            saved_blocks += 1
                            
                except Exception as e:
                    print(f"      ⚠️  Could not save block {i}: {e}")
                    continue
            
            print(f"    ✅ Saved {saved_blocks} individual blocks to: {blocks_dir}")
            return True
            
        except Exception as e:
            print(f"    ❌ Individual blocks saving failed: {e}")
            return False
    
    def create_analysis_dashboard(self, image, layout_blocks, output_dir, base_name):
        """Create comprehensive analysis dashboard"""
        try:
            print("  📊 Creating analysis dashboard...")
            
            # Extract block information
            block_types = []
            confidences = []
            areas = []
            
            for block in layout_blocks:
                if hasattr(block, 'type'):
                    block_types.append(block.type)
                    confidences.append(getattr(block, 'score', 0))
                    x1, y1, x2, y2 = block.coordinates
                    areas.append((x2 - x1) * (y2 - y1))
                else:
                    block_types.append(block['type'])
                    confidences.append(block.get('confidence', 0))
                    coords = block['coordinates']
                    areas.append(coords['width'] * coords['height'])
            
            # Count block types
            type_counts = {}
            for block_type in block_types:
                type_counts[block_type] = type_counts.get(block_type, 0) + 1
            
            # Create dashboard
            fig = plt.figure(figsize=(20, 12))
            
            # Main annotated image
            ax_main = plt.subplot(2, 3, (1, 4))
            ax_main.imshow(image)
            
            for i, block in enumerate(layout_blocks):
                if hasattr(block, 'coordinates'):
                    x1, y1, x2, y2 = block.coordinates
                    block_type = block.type
                else:
                    coords = block['coordinates']
                    x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                    block_type = block['type']
                
                width = x2 - x1
                height = y2 - y1
                color = self.colors.get(block_type, '#666666')
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax_main.add_patch(rect)
                
                ax_main.text(x1, y1-5, f"{i}", fontsize=8, color='white', weight='bold',
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor=color))
            
            ax_main.set_title(f"Layout Analysis Dashboard: {base_name}", fontsize=16)
            ax_main.axis('off')
            
            # Block type distribution
            if type_counts:
                ax_counts = plt.subplot(2, 3, 2)
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                colors_list = [self.colors.get(t, '#666666') for t in types]
                
                bars = ax_counts.bar(types, counts, color=colors_list, alpha=0.7)
                ax_counts.set_title("Block Type Distribution")
                ax_counts.set_ylabel("Count")
                ax_counts.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, count in zip(bars, counts):
                    ax_counts.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                 str(count), ha='center', va='bottom')
            
            # Confidence distribution
            ax_conf = plt.subplot(2, 3, 3)
            if confidences and any(c > 0 for c in confidences):
                valid_conf = [c for c in confidences if c > 0]
                ax_conf.hist(valid_conf, bins=min(10, len(valid_conf)), alpha=0.7, color='skyblue', edgecolor='black')
                ax_conf.set_title("Confidence Score Distribution")
                ax_conf.set_xlabel("Confidence")
                ax_conf.set_ylabel("Frequency")
                if valid_conf:
                    ax_conf.axvline(np.mean(valid_conf), color='red', linestyle='--', 
                                   label=f'Mean: {np.mean(valid_conf):.2f}')
                    ax_conf.legend()
            else:
                ax_conf.text(0.5, 0.5, 'No confidence data', ha='center', va='center', transform=ax_conf.transAxes)
                ax_conf.set_title("Confidence Score Distribution")
            
            # Block areas
            ax_size = plt.subplot(2, 3, 5)
            if areas:
                ax_size.scatter(range(len(areas)), areas, alpha=0.6, c=[self.colors.get(bt, '#666666') for bt in block_types])
                ax_size.set_title("Block Size Distribution")
                ax_size.set_xlabel("Block Index")
                ax_size.set_ylabel("Area (pixels)")
            
            # Summary statistics
            ax_stats = plt.subplot(2, 3, 6)
            ax_stats.axis('off')
            
            stats_text = f"""
ANALYSIS SUMMARY
================
Total Blocks: {len(layout_blocks)}
Image Size: {image.shape[1]} x {image.shape[0]}

BLOCK TYPES:
"""
            
            for block_type, count in type_counts.items():
                stats_text += f"\n• {block_type}: {count}"
            
            if confidences and any(c > 0 for c in confidences):
                valid_conf = [c for c in confidences if c > 0]
                stats_text += f"\n\nCONFIDENCE STATS:"
                stats_text += f"\nMean: {np.mean(valid_conf):.3f}"
                stats_text += f"\nMin: {min(valid_conf):.3f}"
                stats_text += f"\nMax: {max(valid_conf):.3f}"
            
            ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"{base_name}_dashboard.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    ✅ Dashboard saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"    ❌ Dashboard creation failed: {e}")
            return False

def enhance_existing_analysis_fixed(json_file_path, original_image_path):
    """Enhanced version that handles the MockLayoutBlock error"""
    
    # Load existing JSON results
    with open(json_file_path, 'r') as f:
        analysis_data = json.load(f)
    
    # Create fixed layout blocks from JSON data
    layout_blocks = []
    for block_data in analysis_data.get('layout_blocks', []):
        layout_blocks.append(FixedMockLayoutBlock(block_data))
    
    # Create visualizations
    visualizer = FixedVisualizationEnhancer()
    output_dir = Path(json_file_path).parent
    
    print(f"🎨 Creating enhanced visualizations with {len(layout_blocks)} blocks...")
    visualizer.create_full_visualization(original_image_path, layout_blocks, output_dir)
    
    print(f"✅ Enhanced visualizations complete!")

# Example usage
if __name__ == "__main__":
    # Update these paths with your actual files
    json_file = "comprehensive_output/sample1_analysis.json"
    original_image = "images/sample1.jpg"
    
    if os.path.exists(json_file) and os.path.exists(original_image):
        enhance_existing_analysis_fixed(json_file, original_image)
    else:
        print("❌ Please update the file paths:")
        print(f"JSON file: {json_file}")
        print(f"Image file: {original_image}")
        print("\nUpdate the paths in the script and run again.")