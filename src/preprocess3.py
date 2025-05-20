import cv2
import numpy as np
from PIL import Image
import pytesseract

class Preprocess:
    def __init__(self, debug=False):
        self.debug = debug
       
    def log(self, name, image):
        if self.debug:
            max_width=1000
            max_height=700
            h, w = image.shape[:2]
            if w > max_width or h > max_height:
                scale = min(max_width / w, max_height / h)
                img = cv2.resize(image, (int(w * scale), int(h * scale)))
            cv2.imshow(name, img)
            cv2.waitKey(0)
                        

    def preprocess(self, image_path):
        # Load and convert to grayscale
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.log("Gray", gray)

        # Noise removal with bilateral filter
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        self.log("Filtered", filtered)

        # Adaptive thresholding for uneven illumination
        thresh = cv2.adaptiveThreshold(
            filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 10
        )
        self.log("Threshold", thresh)

        # Morphological operations to clean noise
        kernel = np.ones((2,2), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        self.log("Opening", opening)

        # Deskew the image
        coords = np.column_stack(np.where(opening > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = opening.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(opening, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        self.log("Deskewed", deskewed)

        # Invert back to white background
        result = cv2.bitwise_not(deskewed)
        self.log("Final Preprocessed", result)

        return result

    def extract_text(self, processed_image):
        pil_img = Image.fromarray(processed_image)
        return pytesseract.image_to_string(pil_img)

# Example usage
# pre = Preprocess(debug=True)
# result_img = pre.preprocess("path/to/image.jpg")
# text = pre.extract_text(result_img)
# print(text)
