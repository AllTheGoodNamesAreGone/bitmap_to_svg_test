#!/usr/bin/env python3
"""
Final test to verify PaddleOCR is working properly
"""

def test_all_imports():
    """Test all required imports"""
    print("Testing all imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"✗ OpenCV failed: {e}")
        return False
    
    try:
        import paddle
        print(f"✓ PaddlePaddle imported")
    except Exception as e:
        print(f"✗ PaddlePaddle failed: {e}")
        return False
    
    try:
        from paddleocr import PaddleOCR, PPStructure
        print(f"✓ PaddleOCR imported")
    except Exception as e:
        print(f"✗ PaddleOCR failed: {e}")
        return False
    
    return True

def test_paddleocr_functionality():
    """Test actual PaddleOCR functionality"""
    print("\nTesting PaddleOCR functionality...")
    
    try:
        from paddleocr import PPStructure
        
        # Create structure engine
        print("Creating PPStructure engine...")
        engine = PPStructure(show_log=False, lang='en')
        print("✓ PPStructure engine created successfully")
        
        # Create a simple test image
        import numpy as np
        test_img = np.ones((200, 300, 3), dtype=np.uint8) * 255
        test_img[50:100, 50:250] = 0  # Black rectangle
        
        print("Running structure analysis on test image...")
        result = engine(test_img)
        print(f"✓ Analysis completed! Found {len(result)} elements")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def show_version_info():
    """Show version information for debugging"""
    print("\n" + "="*50)
    print("VERSION INFORMATION")
    print("="*50)
    
    import sys
    print(f"Python: {sys.version}")
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except:
        print("NumPy: Not available")
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except:
        print("OpenCV: Not available")
    
    try:
        import paddle
        print(f"PaddlePaddle: Available")
    except:
        print("PaddlePaddle: Not available")
    
    try:
        import paddleocr
        print(f"PaddleOCR: Available")
    except:
        print("PaddleOCR: Not available")

if __name__ == "__main__":
    print("Final PaddleOCR Installation Test")
    print("=" * 40)
    
    show_version_info()
    
    print("\n" + "="*40)
    if test_all_imports():
        print("\n✓ All imports successful!")
        
        if test_paddleocr_functionality():
            print("\n🎉 EVERYTHING IS WORKING!")
            print("\nYou can now:")
            print("1. Update your question paper image path")
            print("2. Run: python src/paddle.py") 
            print("3. Analyze your question papers!")
        else:
            print("\n⚠️  Imports work but functionality test failed")
            print("This might be due to model downloads on first run")
            print("Try running your main analysis script")
    else:
        print("\n❌ Import test failed")
        print("Try the compatible versions installation")