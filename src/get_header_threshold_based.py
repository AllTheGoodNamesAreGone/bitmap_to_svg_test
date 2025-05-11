import cv2
import pytesseract

#simple threshold based header separation
#it simply separates based on a predefined ratio, nothing more, works if all documents have same proportions 


def get_header_threshold_based(img,gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    h, w = img.shape[:2]
    boxes = pytesseract.image_to_boxes(img)
    # Process boxes to find the header (e.g., based on y-coordinate)
    header_threshold = h * 0.4  # Define a threshold for header position
    header = []

    for box in boxes.splitlines():
        b = box.split()
        x, y, _, _ = map(int, b[1:5])
        if y < header_threshold:  # y-coordinate near the top (header)
            header.append(b)

    header_img = img[0:int(header_threshold), :]
    body_img = img[int(header_threshold):, :]

    return header_img, body_img