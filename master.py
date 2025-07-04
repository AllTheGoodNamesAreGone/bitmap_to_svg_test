#!/usr/bin/env python3
"""
Master Analysis Pipeline
Runs all question paper analysis scripts in sequence and organizes outputs
"""

import os
import sys
import shutil
import json
from pathlib import Path
import subprocess
import time

class MasterAnalysisPipeline:
    def __init__(self, input_image, output_base_dir):
        """
        Initialize the master pipeline
        
        Args:
            input_image: Path to the question paper image
            output_base_dir: Base directory for all outputs
        """
        self.input_image = Path(input_image)
        self.output_base_dir = Path(output_base_dir)
        self.image_name = self.input_image.stem  # e.g., "sample1"
        
        # Ensure input image exists
        if not self.input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")
        
        # Create base output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output directories for each script
        self.script_outputs = {
            'advanced_analysis': self.output_base_dir / '01_advanced_analysis',
            'strategy_comparison': self.output_base_dir / '02_strategy_comparison', 
            'hybrid_svg_detection': self.output_base_dir / '03_hybrid_svg_detection',
            'table_parsing': self.output_base_dir / '04_table_parsing',
            'svg_creation': self.output_base_dir / '05_svg_creation',
            'visualization_enhancement': self.output_base_dir / '06_visualization_enhancement'
        }
        
        # Create all output directories
        for dir_path in self.script_outputs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Master Pipeline initialized")
        print(f"📁 Input: {self.input_image}")
        print(f"📂 Output base: {self.output_base_dir}")
    
    def run_full_pipeline(self):
        """Run the complete analysis pipeline"""
        
        print(f"\n{'='*60}")
        print(f"STARTING MASTER ANALYSIS PIPELINE")
        print(f"{'='*60}")
        
        start_time = time.time()
        results = {}
        
        # Step 1: Advanced Analysis (Script 1)
        print(f"\n{'='*50}")
        print(f"STEP 1: ADVANCED ANALYSIS")
        print(f"{'='*50}")
        results['advanced_analysis'] = self.run_advanced_analysis()
        
        # Step 2: Strategy Comparison (Script 2)
        print(f"\n{'='*50}")
        print(f"STEP 2: STRATEGY COMPARISON")
        print(f"{'='*50}")
        results['strategy_comparison'] = self.run_strategy_comparison()
        
        # Step 3: Hybrid SVG Detection (Script 3)
        print(f"\n{'='*50}")
        print(f"STEP 3: HYBRID SVG DETECTION")
        print(f"{'='*50}")
        results['hybrid_svg_detection'] = self.run_hybrid_svg_detection()
        
        # Step 4: Table Parsing
        print(f"\n{'='*50}")
        print(f"STEP 4: TABLE PARSING")
        print(f"{'='*50}")
        results['table_parsing'] = self.run_table_parsing()
        
        # Step 5: SVG Creation (depends on Step 3)
        print(f"\n{'='*50}")
        print(f"STEP 5: SVG CREATION")
        print(f"{'='*50}")
        results['svg_creation'] = self.run_svg_creation(results.get('hybrid_svg_detection'))
        
        # Step 6: Visualization Enhancement (depends on Step 1)
        print(f"\n{'='*50}")
        print(f"STEP 6: VISUALIZATION ENHANCEMENT")
        print(f"{'='*50}")
        results['visualization_enhancement'] = self.run_visualization_enhancement(results.get('advanced_analysis'))
        
        # Generate summary report
        total_time = time.time() - start_time
        self.generate_pipeline_summary(results, total_time)
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE! ✅")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Results saved to: {self.output_base_dir}")
        print(f"{'='*60}")
        
        return results
    
    def run_advanced_analysis(self):
        """Run Script 1: Advanced Analysis"""
        try:
            # Add src directory to path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            # Import and run advanced analysis
            from advanced import AdvancedQuestionPaperAnalyzer
            
            analyzer = AdvancedQuestionPaperAnalyzer()
            results = analyzer.analyze_question_paper(
                str(self.input_image),
                extract_text=True,
                analyze_tables=True
            )
            
            # Save to designated directory
            output_dir = self.script_outputs['advanced_analysis']
            analyzer.save_comprehensive_results(results, output_dir)
            
            print(f"SUCCESS Advanced analysis complete")
            print(f"   Blocks detected: {len(results['layout_blocks'])}")
            print(f"   Tables detected: {len(results['tables'])}")
            print(f"   Saved to: {output_dir}")
            
            return {
                'success': True,
                'blocks_count': len(results['layout_blocks']),
                'tables_count': len(results['tables']),
                'analysis_file': str(output_dir / f"{self.image_name}_analysis.json"),
                'summary_file': str(output_dir / f"{self.image_name}_summary.txt")
            }
            
        except Exception as e:
            print(f"FAILED Advanced analysis failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_strategy_comparison(self):
        """Run Script 2: Strategy Comparison"""
        try:
            # Add src directory to path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                
            from advanced2 import ImprovedQuestionPaperAnalyzer
            
            analyzer = ImprovedQuestionPaperAnalyzer()
            results = analyzer.analyze_with_multiple_strategies(str(self.input_image))
            
            # Move generated files to designated directory
            output_dir = self.script_outputs['strategy_comparison']
            self.move_generated_files(
                patterns=[
                    f"{self.image_name}_strategy_comparison.png",
                    f"{self.image_name}_detailed_results.json",
                    f"{self.image_name}_enhanced_detection.png"
                ],
                destination=output_dir
            )
            
            # Get best strategy info
            best_strategy = analyzer.select_best_strategy(results['strategies'])
            best_count = results['strategies'][best_strategy]['count']
            
            print(f"✅ Strategy comparison complete")
            print(f"   🎯 Best strategy: {best_strategy}")
            print(f"   📊 Best count: {best_count} blocks")
            print(f"   📁 Saved to: {output_dir}")
            
            return {
                'success': True,
                'best_strategy': best_strategy,
                'best_count': best_count,
                'strategies_tested': len(results['strategies']),
                'results_file': output_dir / f"{self.image_name}_detailed_results.json"
            }
            
        except Exception as e:
            print(f"❌ Strategy comparison failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_hybrid_svg_detection(self):
        """Run Script 3: Hybrid SVG Detection"""
        try:
            # Add src directory to path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                
            from advanced3 import HybridSVGDetectionSystem
            
            analyzer = HybridSVGDetectionSystem()
            elements, svg_structure = analyzer.analyze_for_svg_reconstruction(str(self.input_image))
            
            # Move generated files to designated directory
            output_dir = self.script_outputs['hybrid_svg_detection']
            self.move_generated_files(
                patterns=[
                    f"{self.image_name}_comprehensive_svg_analysis.png",
                    f"{self.image_name}_svg_ready_complete.json",
                    f"{self.image_name}_svg_structure.json"
                ],
                destination=output_dir
            )
            
            print(f"✅ Hybrid SVG detection complete")
            print(f"   📊 Elements detected: {len(elements)}")
            print(f"   🎨 SVG groups: {len(svg_structure['groups'])}")
            print(f"   📁 Saved to: {output_dir}")
            
            return {
                'success': True,
                'elements_count': len(elements),
                'svg_groups': len(svg_structure['groups']),
                'svg_ready_file': output_dir / f"{self.image_name}_svg_ready_complete.json",
                'svg_structure_file': output_dir / f"{self.image_name}_svg_structure.json"
            }
            
        except Exception as e:
            print(f"❌ Hybrid SVG detection failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_table_parsing(self):
        """Run Table Parser"""
        try:
            # Add src directory to path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                
            from tableparser import QuestionPaperTableParser
            
            parser = QuestionPaperTableParser()
            cells, svg_structure = parser.parse_question_paper_table(str(self.input_image))
            
            # Move generated files to designated directory
            output_dir = self.script_outputs['table_parsing']
            self.move_generated_files(
                patterns=[
                    f"{self.image_name}_table_parsing_analysis.png",
                    f"{self.image_name}_table_parsing_complete.json",
                    f"{self.image_name}_table_svg_structure.json"
                ],
                destination=output_dir
            )
            
            table_info = svg_structure['table_structure']
            
            print(f"✅ Table parsing complete")
            print(f"   📊 Cells detected: {len(cells)}")
            print(f"   📐 Grid size: {table_info['rows']}x{table_info['cols']}")
            print(f"   📁 Saved to: {output_dir}")
            
            return {
                'success': True,
                'cells_count': len(cells),
                'grid_rows': table_info['rows'],
                'grid_cols': table_info['cols'],
                'complete_file': output_dir / f"{self.image_name}_table_parsing_complete.json"
            }
            
        except Exception as e:
            print(f"❌ Table parsing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_svg_creation(self, hybrid_results):
        """Run SVG Creator (depends on hybrid detection)"""
        try:
            if not hybrid_results or not hybrid_results.get('success'):
                print(f"⚠️  Skipping SVG creation - hybrid detection failed")
                return {'success': False, 'error': 'Missing hybrid detection results'}
            
            # Add src directory to path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                
            from create_svg import FixedSVGCreator
            
            creator = FixedSVGCreator()
            
            # Use SVG-ready file from hybrid detection
            svg_ready_file = hybrid_results['svg_ready_file']
            
            svg_path, data = creator.create_svg_from_detection(
                str(svg_ready_file),
                str(self.input_image)
            )
            
            # Move generated files to designated directory
            output_dir = self.script_outputs['svg_creation']
            self.move_generated_files(
                patterns=[
                    f"{self.image_name}_reconstructed.svg",
                    f"{self.image_name}_reconstructed_data.json",
                    f"{self.image_name}_reconstructed_preview.html"
                ],
                destination=output_dir
            )
            
            print(f"✅ SVG creation complete")
            print(f"   📄 SVG file: {output_dir / f'{self.image_name}_reconstructed.svg'}")
            print(f"   🌐 Preview: {output_dir / f'{self.image_name}_reconstructed_preview.html'}")
            print(f"   📁 Saved to: {output_dir}")
            
            return {
                'success': True,
                'svg_file': output_dir / f"{self.image_name}_reconstructed.svg",
                'preview_file': output_dir / f"{self.image_name}_reconstructed_preview.html",
                'elements_processed': data['total_elements']
            }
            
        except Exception as e:
            print(f"❌ SVG creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_visualization_enhancement(self, advanced_results):
        """Run Visualization Enhancer (depends on advanced analysis)"""
        try:
            if not advanced_results or not advanced_results.get('success'):
                print(f"⚠️  Skipping visualization enhancement - advanced analysis failed")
                return {'success': False, 'error': 'Missing advanced analysis results'}
            
            # Add src directory to path
            src_path = Path(__file__).parent / 'src'
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
                
            from visualizer import enhance_existing_analysis_fixed
            
            # Use analysis file from advanced analysis
            analysis_file = advanced_results['analysis_file']
            
            # Ensure the file exists before proceeding
            if not Path(analysis_file).exists():
                print(f"    ⚠️  Analysis file not found: {analysis_file}")
                # Try to find the file in the expected location
                expected_file = self.script_outputs['advanced_analysis'] / f"{self.image_name}_analysis.json"
                if expected_file.exists():
                    analysis_file = expected_file
                    print(f"    🔍 Found analysis file at: {analysis_file}")
                else:
                    return {'success': False, 'error': f'Analysis file not found: {analysis_file}'}
            
            # Convert to absolute paths before changing directory
            analysis_file_abs = Path(analysis_file).resolve()
            image_file_abs = Path(self.input_image).resolve()
            
            # Temporarily set output to our directory
            original_cwd = os.getcwd()
            output_dir = self.script_outputs['visualization_enhancement']
            os.chdir(str(output_dir))
            
            try:
                enhance_existing_analysis_fixed(str(analysis_file_abs), str(image_file_abs))
            finally:
                os.chdir(original_cwd)
            
            # Move any files from root directory
            self.move_generated_files(
                patterns=[
                    f"{self.image_name}_dashboard.png",
                    f"{self.image_name}_layoutparser_viz.png",
                    f"{self.image_name}_matplotlib_viz.png",
                    f"{self.image_name}_manual_viz.png"
                ],
                destination=output_dir,
                check_subdirs=True
            )
            
            # Count blocks folder if it exists
            blocks_folder = output_dir / f"{self.image_name}_blocks"
            blocks_count = len(list(blocks_folder.glob("*.png"))) if blocks_folder.exists() else 0
            
            print(f"✅ Visualization enhancement complete")
            print(f"   🎨 Visualizations created: 4 types")
            print(f"   📷 Individual blocks: {blocks_count}")
            print(f"   📁 Saved to: {output_dir}")
            
            return {
                'success': True,
                'visualizations_count': 4,
                'individual_blocks': blocks_count,
                'dashboard_file': output_dir / f"{self.image_name}_dashboard.png"
            }
            
        except Exception as e:
            print(f"FAILED Visualization enhancement failed: {e}")
            print(f"    Debug info:")
            print(f"    Analysis file: {advanced_results.get('analysis_file', 'N/A')}")
            print(f"    File exists: {Path(advanced_results.get('analysis_file', '')).exists() if advanced_results.get('analysis_file') else False}")
            return {'success': False, 'error': str(e)}
    
    def move_generated_files(self, patterns, destination, check_subdirs=False):
        """Move generated files from current directory to destination"""
        moved_count = 0
        
        for pattern in patterns:
            # Check in current directory
            current_files = list(Path('.').glob(pattern))
            
            # Check in subdirectories if requested
            if check_subdirs and not current_files:
                current_files = list(Path('.').rglob(pattern))
            
            for file_path in current_files:
                try:
                    dest_path = destination / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    moved_count += 1
                except Exception as e:
                    print(f"⚠️  Could not move {file_path}: {e}")
        
        return moved_count
    
    def generate_pipeline_summary(self, results, total_time):
        """Generate comprehensive summary report"""
        
        summary_file = self.output_base_dir / "PIPELINE_SUMMARY.txt"
        
        summary_content = f"""
MASTER ANALYSIS PIPELINE SUMMARY
{'='*50}

Image: {self.input_image.name}
Processing Time: {total_time:.1f} seconds
Output Directory: {self.output_base_dir}

SCRIPT RESULTS:
{'='*30}

1. ADVANCED ANALYSIS:
   Status: {('SUCCESS' if results.get('advanced_analysis', {}).get('success') else 'FAILED')}
   Blocks: {results.get('advanced_analysis', {}).get('blocks_count', 'N/A')}
   Tables: {results.get('advanced_analysis', {}).get('tables_count', 'N/A')}

2. STRATEGY COMPARISON:
   Status: {('SUCCESS' if results.get('strategy_comparison', {}).get('success') else 'FAILED')}
   Best Strategy: {results.get('strategy_comparison', {}).get('best_strategy', 'N/A')}
   Best Count: {results.get('strategy_comparison', {}).get('best_count', 'N/A')}

3. HYBRID SVG DETECTION:
   Status: {('SUCCESS' if results.get('hybrid_svg_detection', {}).get('success') else 'FAILED')}
   Elements: {results.get('hybrid_svg_detection', {}).get('elements_count', 'N/A')}
   SVG Groups: {results.get('hybrid_svg_detection', {}).get('svg_groups', 'N/A')}

4. TABLE PARSING:
   Status: {('SUCCESS' if results.get('table_parsing', {}).get('success') else 'FAILED')}
   Cells: {results.get('table_parsing', {}).get('cells_count', 'N/A')}
   Grid: {results.get('table_parsing', {}).get('grid_rows', 'N/A')}x{results.get('table_parsing', {}).get('grid_cols', 'N/A')}

5. SVG CREATION:
   Status: {('SUCCESS' if results.get('svg_creation', {}).get('success') else 'FAILED')}
   Elements Processed: {results.get('svg_creation', {}).get('elements_processed', 'N/A')}

6. VISUALIZATION ENHANCEMENT:
   Status: {('SUCCESS' if results.get('visualization_enhancement', {}).get('success') else 'FAILED')}
   Visualizations: {results.get('visualization_enhancement', {}).get('visualizations_count', 'N/A')}
   Individual Blocks: {results.get('visualization_enhancement', {}).get('individual_blocks', 'N/A')}

OUTPUT DIRECTORIES:
{'='*30}
"""
        
        for name, path in self.script_outputs.items():
            files_count = len(list(path.rglob("*"))) if path.exists() else 0
            summary_content += f"{name}: {path} ({files_count} files)\n"
        
        summary_content += f"""
QUICK ACCESS FILES:
{'='*30}
Final SVG: {self.script_outputs['svg_creation'] / f'{self.image_name}_reconstructed.svg'}
HTML Preview: {self.script_outputs['svg_creation'] / f'{self.image_name}_reconstructed_preview.html'}
Analysis Dashboard: {self.script_outputs['visualization_enhancement'] / f'{self.image_name}_dashboard.png'}
Table Analysis: {self.script_outputs['table_parsing'] / f'{self.image_name}_table_parsing_analysis.png'}

ERRORS (if any):
{'='*30}
"""
        
        for script_name, result in results.items():
            if not result.get('success'):
                summary_content += f"{script_name}: {result.get('error', 'Unknown error')}\n"
        
        # Save summary with UTF-8 encoding to handle special characters
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📋 Pipeline summary saved: {summary_file}")
        
        # Debug: List all created files
        print(f"\n🔍 DEBUG: Files created in each directory:")
        for name, path in self.script_outputs.items():
            if path.exists():
                files = list(path.glob("*"))
                print(f"   {name}: {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    print(f"     - {file.name}")
                if len(files) > 3:
                    print(f"     ... and {len(files)-3} more")
            else:
                print(f"   {name}: Directory not found")


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Master Analysis Pipeline")
    parser.add_argument("input_image", help="Path to input question paper image")
    parser.add_argument("output_dir", help="Base output directory for all results")
    parser.add_argument("--skip", nargs="+", choices=['advanced', 'strategy', 'hybrid', 'table', 'svg', 'viz'], 
                       help="Skip specific steps", default=[])
    
    args = parser.parse_args()
    
    try:
        pipeline = MasterAnalysisPipeline(args.input_image, args.output_dir)
        results = pipeline.run_full_pipeline()
        
        print(f"\n🎉 SUCCESS! All results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        # Default example
        input_image = "images/sample1.jpg"
        output_dir = "pipeline_results"
        
        print("🚀 Running with default parameters...")
        print(f"📷 Input: {input_image}")
        print(f"📂 Output: {output_dir}")
        
        pipeline = MasterAnalysisPipeline(input_image, output_dir)
        results = pipeline.run_full_pipeline()
    else:
        # Command line usage
        main()