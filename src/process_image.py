#ORIGINAL PROCESSING FUNCTION - TO FIND BOUNDING LAYOUT BOXES

# WORKS WELL FOR TABULAR USING RETR, BUT NOT SO WELL FOR TEXT BOXES - CAN FIND INDIVIDUAL CHARACTERS THOUGH
#use this maybe for tabular body section

import cv2
from display_image import display_image
import os

#BODY PROCESSING -----------------------------

def process_body(image_path, output_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #display_image(gray, "step 1 - gray")


    #finding header and body distinction
    #header,body = get_header_threshold_based(img, gray)
    #display_image(header, "Header")
    #display_image(body, "body")

    # Apply adaptive thresholding
    #ALTERNATIVE OPTION - ADAPTIVE THRESH GAUSSIAN, ORIGINAL PARAMETERS - 15,10
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5) 
    
    #display_image(thresh, "step 2 - after thresh")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    #display_image(dilated, "step 3 - dilated")
    # Find contours (connected components)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #display_image(contours, "step 4 - contours")

    # List to store bounding boxes
    boxes = []

    # Loop through contours to get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #print(f"Box: x={x}, y={y}, w={w}, h={h}")
        """
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 10 and w > 20 and h > 15:
            boxes.append((x, y, w, h))
        """
        if ( w > 80 and h > 20):  # Filter out noise based on box size (adjust as needed)
            boxes.append((x, y, w, h))
        

    # Draw bounding boxes on the original image
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    # Save the output image with bounding boxes
    #display_image(img)
    cv2.imwrite(output_path, img)
    print(f"Bounding boxes drawn and saved to {output_path}")



# Function to process image and find bounding boxes
def process_image(image_path, output_path):
    # Read the image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    
    # Check if image is loaded successfully
    if img is None:
        print(f"Error: Could not read the image at {image_path}")
        return
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #display_image(gray, "step 1 - gray")


    #finding header and body distinction
    #header,body = get_header_threshold_based(img, gray)
    #display_image(header, "Header")
    #display_image(body, "body")

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    #display_image(thresh, "step 2 - after thresh")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    #display_image(dilated, "step 3 - dilated")
    # Find contours (connected components)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #display_image(contours, "step 4 - contours")

    # List to store bounding boxes
    boxes = []

    # Loop through contours to get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #print(f"Box: x={x}, y={y}, w={w}, h={h}")
        if ( w > 30 and h > 20):  # Filter out noise based on box size (adjust as needed)
            boxes.append((x, y, w, h))

    # Draw bounding boxes on the original image
    for x, y, w, h in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    # Save the output image with bounding boxes
    display_image(img)
    cv2.imwrite(output_path, img)
    print(f"Bounding boxes drawn and saved to {output_path}")
