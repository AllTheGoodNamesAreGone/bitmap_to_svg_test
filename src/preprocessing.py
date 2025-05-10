# src/preprocessing.py
import cv2
import numpy as np
from .utils import load_image, save_processed_image

class DocumentPreprocessor:
    def __init__(self):
        pass
    
    def preprocess(self, image_path, debug=False):
        """Main preprocessing pipeline for document images."""
        # Load the image
        original, _ = load_image(image_path)
        
        # Apply preprocessing steps
        gray = self.convert_to_grayscale(original)
        denoised = self.remove_noise(gray)
        binary = self.binarize(denoised)
        deskewed = self.deskew(binary)
        
        # Save intermediate results if debug is True
        if debug:
            save_processed_image(gray, "1_grayscale.jpg")
            save_processed_image(denoised, "2_denoised.jpg")
            save_processed_image(binary, "3_binary.jpg")
            save_processed_image(deskewed, "4_deskewed.jpg")
        
        return deskewed
    
    def convert_to_grayscale(self, image):
        """Convert image to grayscale if it's not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def remove_noise(self, image):
        """Remove noise using Gaussian blur."""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def binarize(self, image):
        """Convert to binary using adaptive thresholding."""
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    
    def deskew(self, image):
        """Correct skew in the document."""
        # Find all non-zero points
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
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
            return rotated
        
        return image