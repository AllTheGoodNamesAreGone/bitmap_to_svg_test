import cv2
from display_image import display_image


#This doesnt work that well, but is mainly for line detection - detects straight lines based on parameters and gaps, doesnt work well for slight ccurves

def get_header_hough_based (img, gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    edges = cv2.Canny(blurred, 30, 100) 
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=300, maxLineGap=10)
    debug_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10 and abs(x2 - x1) > 100:
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_image(debug_img)

    # Loop over lines and find the lowest y of a horizontal line
    header_end = float('inf')
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 10 and abs(x2 - x1) > 100:  # horizontal line
            header_end = min(header_end, y1)

    # Now split using the first major horizontal line
    header = img[:header_end, :]
    body = img[header_end:, :]

    return header,body