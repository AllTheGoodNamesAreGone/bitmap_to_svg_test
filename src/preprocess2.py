import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

class DocumentPreprocessor:
    """Enhanced document image preprocessing pipeline with visualization capabilities"""
    
    def __init__(self, output_dir=None):
        """
        Initialize the document preprocessor
        
        Args:
            output_dir (str, optional): Directory to save intermediate images
        """
        self.output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def load_image(self, image_path):
        """
        Load an image from the specified path
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
            str: Filename without extension
        """
        # Check if the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Get filename without extension for saving intermediate results
        filename = Path(image_path).stem
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        return image, filename
    
    def save_or_display(self, image, name, filename=None, display=False, save=False):
        """
        Save and/or display an image
        
        Args:
            image (numpy.ndarray): Image to save/display
            name (str): Name of processing step
            filename (str, optional): Base filename for saving
            display (bool): Whether to display the image
            save (bool): Whether to save the image
        """
        if save and self.output_dir and filename:
            output_path = os.path.join(self.output_dir, f"{filename}_{name}.jpg")
            cv2.imwrite(output_path, image)
            
        if display:
            plt.figure(figsize=(10, 10))
            if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale
                plt.imshow(image, cmap='gray')
            else:  # Color (BGR to RGB for display)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(name)
            plt.axis('off')
            plt.show()
    
    def convert_to_grayscale(self, image):
        """
        Convert image to grayscale if it's not already
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def remove_noise(self, image, method='gaussian', kernel_size=5):
        """
        Remove noise from the image using various methods
        
        Args:
            image (numpy.ndarray): Input grayscale image
            method (str): Noise removal method ('gaussian', 'median', 'bilateral')
            kernel_size (int): Size of kernel for filtering
            
        Returns:
            numpy.ndarray: Denoised image
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            raise ValueError(f"Unsupported noise removal method: {method}")
    
    def normalize_illumination(self, image):
        """
        Normalize illumination using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image (numpy.ndarray): Input grayscale image
            
        Returns:
            numpy.ndarray: Image with normalized illumination
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def binarize(self, image, method='adaptive', block_size=11, c=2):
        """
        Convert grayscale image to binary using various thresholding methods
        
        Args:
            image (numpy.ndarray): Input grayscale image
            method (str): Binarization method ('adaptive', 'otsu', 'sauvola')
            block_size (int): Block size for adaptive methods
            c (int): Constant subtracted from mean for adaptive methods
            
        Returns:
            numpy.ndarray: Binary image
        """
        if method == 'adaptive':
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
        elif method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'sauvola':
            # Simplified Sauvola implementation using adaptive threshold
            # True Sauvola requires more complex implementation
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, block_size, c
            )
        else:
            raise ValueError(f"Unsupported binarization method: {method}")
    
    def deskew(self, image, max_angle=45):
        """
        Correct skew in the document image
        
        Args:
            image (numpy.ndarray): Input binary image
            max_angle (float): Maximum angle to correct (degrees)
            
        Returns:
            numpy.ndarray: Deskewed image
            float: Detected skew angle
        """
        # Find all non-zero points
        coords = np.column_stack(np.where(image > 0))
        
        # Skip if not enough points
        if len(coords) < 20:
            return image, 0.0
        
        # Find minimum area rectangle
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Limit correction to max_angle
        if abs(angle) > max_angle:
            angle = max_angle if angle > 0 else -max_angle
            
        # Rotate the image to deskew it if angle is significant
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated, angle
        
        return image, angle
    
    def remove_borders(self, image, margin=5):
        """
        Remove dark borders from the image
        
        Args:
            image (numpy.ndarray): Input binary image
            margin (int): Margin to keep around the content
            
        Returns:
            numpy.ndarray: Image with borders removed
        """
        # Find foreground pixels
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) == 0:
            return image
            
        # Find bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add margin
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(image.shape[0] - 1, y_max + margin)
        x_max = min(image.shape[1] - 1, x_max + margin)
        
        # Crop the image
        return image[y_min:y_max, x_min:x_max]
    
    def morphological_operations(self, image, operation='close', kernel_size=3):
        """
        Apply morphological operations to the image
        
        Args:
            image (numpy.ndarray): Input binary image
            operation (str): Operation type ('open', 'close', 'dilate', 'erode')
            kernel_size (int): Size of kernel for morphological operations
            
        Returns:
            numpy.ndarray: Processed image
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'open':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            return cv2.dilate(image, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(image, kernel, iterations=1)
        else:
            raise ValueError(f"Unsupported morphological operation: {operation}")
    
    def preprocess(self, image_path, display=False, save=False, 
                   denoise_method='gaussian', binarize_method='adaptive', 
                   apply_morphology=True, remove_doc_borders=True):
        """
        Main preprocessing pipeline for document images with enhanced options
        
        Args:
            image_path (str): Path to the input image
            display (bool): Whether to display intermediate results
            save (bool): Whether to save intermediate results
            denoise_method (str): Method for denoising ('gaussian', 'median', 'bilateral')
            binarize_method (str): Method for binarization ('adaptive', 'otsu', 'sauvola')
            apply_morphology (bool): Whether to apply morphological operations
            remove_doc_borders (bool): Whether to remove document borders
            
        Returns:
            numpy.ndarray: Preprocessed image
            dict: Dictionary with intermediate results and metadata
        """
        # Dictionary to store intermediate results and metadata
        results = {}
        
        # Load the image
        original, filename = self.load_image(image_path)
        results['original'] = original
        self.save_or_display(original, "original", filename, display, save)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(original)
        results['grayscale'] = gray
        self.save_or_display(gray, "grayscale", filename, display, save)
        
        # Normalize illumination
        normalized = self.normalize_illumination(gray)
        results['normalized'] = normalized
        self.save_or_display(normalized, "normalized", filename, display, save)
        
        # Remove noise
        denoised = self.remove_noise(normalized, method=denoise_method)
        results['denoised'] = denoised
        self.save_or_display(denoised, "denoised", filename, display, save)
        
        # Binarize
        binary = self.binarize(denoised, method=binarize_method)
        results['binary'] = binary
        self.save_or_display(binary, "binary", filename, display, save)
        
        # Apply morphological operations if requested
        if apply_morphology:
            morphed = self.morphological_operations(binary, operation='close')
            results['morphological'] = morphed
            self.save_or_display(morphed, "morphological", filename, display, save)
        else:
            morphed = binary
        
        # Deskew
        deskewed, angle = self.deskew(morphed)
        results['deskewed'] = deskewed
        results['skew_angle'] = angle
        self.save_or_display(deskewed, f"deskewed_{angle:.2f}deg", filename, display, save)
        
        # Remove borders if requested
        if remove_doc_borders:
            final = self.remove_borders(deskewed)
            results['final'] = final
            self.save_or_display(final, "final", filename, display, save)
        else:
            final = deskewed
            results['final'] = final
        
        return final, results


# Example usage
if __name__ == "__main__":
    # Create preprocessor with output directory
    preprocessor = DocumentPreprocessor(output_dir="output_images")
    
    # Process an image with visualization
    image_path = "sample_document.jpg"
    try:
        processed_image, results = preprocessor.preprocess(
            image_path,
            display=True,  # Display intermediate results
            save=True,     # Save intermediate results
            denoise_method='gaussian',
            binarize_method='adaptive',
            apply_morphology=True,
            remove_doc_borders=True
        )
        
        print(f"Preprocessing completed successfully")
        print(f"Detected skew angle: {results['skew_angle']:.2f} degrees")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")