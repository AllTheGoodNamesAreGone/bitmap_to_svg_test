#!/usr/bin/env python3
"""
Improved Question Paper Detection with multiple models and lower thresholds
"""

import layoutparser as lp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import os

class ImprovedQuestionPaperAnalyzer:
    def __init__(self):
        """Initialize with multiple models and aggressive detection"""
        
        # Model 1: General layout with very low threshold
        self.general_model = lp.Detectron2LayoutModel(
            config_path='models\publaynet_mask_rcnn_R_50_FPN_3x/config.yml',
            model_path='models\publaynet_mask_rcnn_R_50_FPN_3x/model_final.pth',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3],  # Very low threshold
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        )
        
        # Model 2: Table-specific detection
        try:
            self.table_model = lp.Detectron2LayoutModel(
                config_path='models/tableBank_faster_rcnn_R_50_FPN_3x/config.yaml',
                model_path='models/tableBank_faster_rcnn_R_50_FPN_3x/model_final.pth',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
                label_map={0: "Table"}
            )
            self.has_table_model = True
        except:
            print("⚠️  TableBank model not available, using general model only")
            self.has_table_model = False
        
        print("✅ Improved analyzer initialized")
    
    def analyze_with_multiple_strategies(self, image_path):
        """Analyze using multiple detection strategies"""
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"🔍 Analyzing {image_path} with multiple strategies...")
        
        results = {
            'image_path': image_path,
            'image_shape': image_rgb.shape,
            'strategies': {}
        }
        
        # Strategy 1: Low threshold general detection
        print("📋 Strategy 1: Low threshold general detection...")
        general_blocks = self.general_model.detect(image_rgb)
        results['strategies']['general_low_threshold'] = {
            'blocks': self.process_blocks(general_blocks),
            'count': len(general_blocks)
        }
        print(f"   Found {len(general_blocks)} blocks")
        
        # Strategy 2: Very low threshold
        print("📋 Strategy 2: Very low threshold detection...")
        self.general_model.extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.1]
        very_low_blocks = self.general_model.detect(image_rgb)
        results['strategies']['very_low_threshold'] = {
            'blocks': self.process_blocks(very_low_blocks),
            'count': len(very_low_blocks)
        }
        print(f"   Found {len(very_low_blocks)} blocks")
        
        # Strategy 3: Table-specific detection
        if self.has_table_model:
            print("📋 Strategy 3: Table-specific detection...")
            table_blocks = self.table_model.detect(image_rgb)
            results['strategies']['table_specific'] = {
                'blocks': self.process_blocks(table_blocks),
                'count': len(table_blocks)
            }
            print(f"   Found {len(table_blocks)} table blocks")
        
        # Strategy 4: Image preprocessing + detection
        print("📋 Strategy 4: Preprocessed image detection...")
        preprocessed_image = self.preprocess_for_detection(image_rgb)
        preprocessed_blocks = self.general_model.detect(preprocessed_image)
        results['strategies']['preprocessed'] = {
            'blocks': self.process_blocks(preprocessed_blocks),
            'count': len(preprocessed_blocks)
        }
        print(f"   Found {len(preprocessed_blocks)} blocks on preprocessed image")
        
        # Strategy 5: Manual grid detection for tables
        print("📋 Strategy 5: Manual table grid detection...")
        manual_blocks = self.detect_table_manually(image_rgb)
        results['strategies']['manual_table'] = {
            'blocks': manual_blocks,
            'count': len(manual_blocks)
        }
        print(f"   Found {len(manual_blocks)} manual table regions")
        
        # Combine and visualize best results
        best_strategy = self.select_best_strategy(results['strategies'])
        print(f"🎯 Best strategy: {best_strategy} with {results['strategies'][best_strategy]['count']} blocks")
        
        # Create comprehensive visualization
        self.create_multi_strategy_visualization(image_rgb, results, image_path)
        
        return results
    
    def preprocess_for_detection(self, image):
        """Preprocess image to enhance detection"""
        # Convert to grayscale and back to enhance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return enhanced_rgb
    
    def detect_table_manually(self, image):
        """Manually detect table regions using edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours (potential table regions)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        manual_blocks = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (must be reasonably large)
            if w > 100 and h > 50:
                manual_blocks.append({
                    'id': f'manual_{i}',
                    'type': 'Table',
                    'confidence': 0.8,
                    'coordinates': {'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h},
                    'dimensions': {'width': w, 'height': h},
                    'area': w * h,
                    'detection_method': 'manual_grid'
                })
        
        return manual_blocks
    
    def process_blocks(self, layout_blocks):
        """Process layout blocks into standard format"""
        processed = []
        
        for i, block in enumerate(layout_blocks):
            processed.append({
                'id': i,
                'type': block.type,
                'confidence': float(block.score),
                'coordinates': {
                    'x1': int(block.coordinates[0]),
                    'y1': int(block.coordinates[1]),
                    'x2': int(block.coordinates[2]),
                    'y2': int(block.coordinates[3])
                },
                'dimensions': {
                    'width': int(block.coordinates[2] - block.coordinates[0]),
                    'height': int(block.coordinates[3] - block.coordinates[1])
                },
                'area': int((block.coordinates[2] - block.coordinates[0]) * 
                           (block.coordinates[3] - block.coordinates[1]))
            })
        
        return processed
    
    def select_best_strategy(self, strategies):
        """Select the strategy that found the most reasonable number of blocks"""
        best_strategy = None
        best_score = 0
        
        for strategy_name, strategy_data in strategies.items():
            count = strategy_data['count']
            
            # Score based on number of blocks (we want more, but not too many)
            if 8 <= count <= 50:  # Reasonable range for a question paper
                score = count
            elif count > 50:  # Too many might be noise
                score = 50 - (count - 50) * 0.5
            else:  # Too few
                score = count * 0.5
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        return best_strategy if best_strategy else list(strategies.keys())[0]
    
    def create_multi_strategy_visualization(self, image, results, image_path):
        """Create visualization comparing all strategies"""
        
        strategies = results['strategies']
        n_strategies = len(strategies)
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis('off')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Plot each strategy
        for i, (strategy_name, strategy_data) in enumerate(strategies.items(), 1):
            if i < len(axes):
                axes[i].imshow(image)
                
                blocks = strategy_data['blocks']
                color = colors[(i-1) % len(colors)]
                
                for block in blocks:
                    coords = block['coordinates']
                    x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
                    width = x2 - x1
                    height = y2 - y1
                    
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor=color, facecolor='none', alpha=0.7
                    )
                    axes[i].add_patch(rect)
                    
                    # Add block number
                    axes[i].text(x1, y1-5, str(block['id']), 
                               fontsize=8, color=color, weight='bold')
                
                axes[i].set_title(f"{strategy_name}\n{len(blocks)} blocks", fontsize=10)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(strategies) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save comparison
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_strategy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Strategy comparison saved: {output_path}")
        
        # Save detailed results
        with open(f"{base_name}_detailed_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Detailed results saved: {base_name}_detailed_results.json")
    
    def create_enhanced_detection_visualization(self, image, best_blocks, image_path):
        """Create detailed visualization of the best detection results"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16))
        
        # Original
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')
        
        # Enhanced detection
        ax2.imshow(image)
        
        # Color map for different types
        type_colors = {
            'Text': '#2E86AB',
            'Title': '#A23B72', 
            'Table': '#F18F01',
            'Figure': '#C73E1D',
            'List': '#6A994E'
        }
        
        for block in best_blocks:
            coords = block['coordinates']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            width = x2 - x1
            height = y2 - y1
            
            color = type_colors.get(block['type'], '#666666')
            
            # Draw thick border
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax2.add_patch(rect)
            
            # Enhanced label
            label = f"{block['id']}: {block['type']}"
            confidence = block.get('confidence', 0)
            conf_text = f"({confidence:.2f})" if confidence > 0 else ""
            
            ax2.text(x1, y1-10, f"{label} {conf_text}", 
                    fontsize=9, color='white', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.9))
        
        ax2.set_title(f"Enhanced Detection: {len(best_blocks)} blocks", fontsize=14)
        ax2.axis('off')
        
        # Add legend
        legend_elements = []
        for block_type in set(block['type'] for block in best_blocks):
            color = type_colors.get(block_type, '#666666')
            legend_elements.append(patches.Patch(color=color, label=block_type))
        
        if legend_elements:
            ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_enhanced_detection.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Enhanced detection saved: {output_path}")

# Usage
if __name__ == "__main__":
    analyzer = ImprovedQuestionPaperAnalyzer()
    
    # Update with your image path
    image_path = "images/sample1.jpg"
    
    if os.path.exists(image_path):
        results = analyzer.analyze_with_multiple_strategies(image_path)
        
        # Get best strategy results
        best_strategy = analyzer.select_best_strategy(results['strategies'])
        best_blocks = results['strategies'][best_strategy]['blocks']
        
        # Create enhanced visualization
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        analyzer.create_enhanced_detection_visualization(image_rgb, best_blocks, image_path)
        
        print(f"\n🎉 Analysis complete!")
        print(f"Best strategy: {best_strategy}")
        print(f"Detected {len(best_blocks)} blocks")
        
        # Show block summary
        type_counts = {}
        for block in best_blocks:
            type_counts[block['type']] = type_counts.get(block['type'], 0) + 1
        
        print("\nBlock type summary:")
        for block_type, count in type_counts.items():
            print(f"  {block_type}: {count}")
            
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable with your actual file path")