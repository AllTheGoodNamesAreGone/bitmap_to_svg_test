import cv2
import numpy as np
import os
import math
from deskew import determine_skew

# --- order_points, perspective_transform, segment_intersection (No changes needed here)
def order_points(pts):
    if not isinstance(pts, np.ndarray) or pts.shape != (4, 2):
        try: pts = np.array(pts, dtype="float32"); assert pts.shape == (4, 2)
        except: print(f"Error: Invalid pts shape for order_points: {getattr(pts, 'shape', 'N/A')}"); return None
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def perspective_transform(image, pts):
    rect = order_points(pts);
    if rect is None: return None
    (tl, tr, br, bl) = rect
    widthA=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2)); widthB=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2)); maxWidth=max(int(widthA),int(widthB))
    heightA=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2)); heightB=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2)); maxHeight=max(int(heightA),int(heightB))
    if maxWidth<=0 or maxHeight<=0: print("Error: Zero dimensions in warp."); return None
    dst=np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")
    M=cv2.getPerspectiveTransform(rect,dst); warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight),flags=cv2.INTER_LANCZOS4)
    return warped

def segment_intersection(seg1, seg2):
    x1,y1,x2,y2 = seg1; x3,y3,x4,y4 = seg2
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);
    if denom==0: return None
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom; u = -((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/denom
    ix=x1+t*(x2-x1); iy=y1+t*(y2-y1); return [int(round(ix)),int(round(iy))]


def preprocess_document_v27_bilateral_smooth(input_path, output_path, debug_folder="debug_images_v27",
                                              apply_post_warp_deskew=True,
                                              adaptive_thresh_blocksize=51,
                                              adaptive_thresh_C=8):
    """
    Uses a Bilateral Filter for intelligent, edge-preserving smoothing (v27).
    Pipeline: Warp -> Deskew -> Create Mask -> Apply to Canvas -> Final Bilateral Smooth
    """
    print(f"--- Starting Preprocessing v27 (Bilateral Smooth) for: {input_path} ---")

    # --- Setup Debug Folder & Save Utility ---
    if not os.path.exists(debug_folder): os.makedirs(debug_folder)
    def save_debug_image(img_data, filename_base):
        if img_data is not None and img_data.size > 0:
            fname = os.path.join(debug_folder, f"{filename_base}.png")
            try: cv2.imwrite(fname, img_data); print(f"Saved Debug Image: {fname}")
            except Exception as e: print(f"Error saving debug {fname}: {e}")

    # --- Step 1: Load, Find Edges, Warp, and Deskew (Proven Method) ---
    img_orig = cv2.imread(input_path)
    if img_orig is None: print(f"Error: Could not read image at {input_path}"); return False
    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    
    # ... (Edge detection, line filtering, and corner finding logic is unchanged) ...
    blurred_for_edges = cv2.GaussianBlur(gray, (7, 7), 0); edged = cv2.Canny(blurred_for_edges, 40, 120)
    lines = cv2.HoughLinesP(edged, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=20)
    if lines is None: print("No lines found, using original gray image."); cv2.imwrite(output_path, gray); return False
    h_orig, w_orig = img_orig.shape[:2]
    horizontal_lines, vertical_lines = [], []
    angle_tolerance_deg=15; min_len=0.1*min(w_orig, h_orig)
    for line in lines:
        x1,y1,x2,y2=line[0]; length=math.sqrt((x2-x1)**2+(y2-y1)**2)
        if length<min_len: continue
        angle_deg=90 if x2-x1==0 else math.degrees(math.atan(abs(y2-y1)/abs(x2-x1)))
        if angle_deg < angle_tolerance_deg: horizontal_lines.append(line[0])
        elif abs(angle_deg - 90) < angle_tolerance_deg: vertical_lines.append(line[0])
    if not horizontal_lines or not vertical_lines: print("Not enough H/V lines found."); cv2.imwrite(output_path, gray); return False
    top_line=min(horizontal_lines,key=lambda l:(l[1]+l[3])/2); bottom_line=max(horizontal_lines,key=lambda l:(l[1]+l[3])/2)
    left_line=min(vertical_lines,key=lambda l:(l[0]+l[2])/2); right_line=max(vertical_lines,key=lambda l:(l[0]+l[2])/2)
    corners=[]; tl=segment_intersection(top_line,left_line); tr=segment_intersection(top_line,right_line); br=segment_intersection(bottom_line,right_line); bl=segment_intersection(bottom_line,left_line)
    if tl: corners.append(tl);
    if tr: corners.append(tr);
    if br: corners.append(br);
    if bl: corners.append(bl);
    
    image_to_process = gray
    if len(corners) == 4:
        warped_color = perspective_transform(img_orig, np.array(corners, dtype="float32"))
        if warped_color is not None: image_to_process = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    if apply_post_warp_deskew:
        try:
            angle = determine_skew(image_to_process)
            if angle is not None and -45 < angle < 45 and abs(angle) > 0.1:
                (h, w) = image_to_process.shape; center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image_to_process = cv2.warpAffine(image_to_process, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255))
        except Exception as e: print(f"Error during deskewing: {e}.")
    save_debug_image(image_to_process, "08_deskewed_gray")

    # --- Step 2: Create High-Contrast Mask ---
    text_mask = cv2.adaptiveThreshold(image_to_process, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_thresh_blocksize, adaptive_thresh_C)
    save_debug_image(text_mask, "09_content_mask")

    # --- Step 3: Apply Mask to White Canvas ---
    h, w = image_to_process.shape; white_canvas = np.ones((h, w), dtype=np.uint8) * 255
    sharp_binary_image = cv2.bitwise_or(white_canvas, white_canvas, mask=text_mask)
    sharp_binary_image = cv2.bitwise_not(sharp_binary_image)
    save_debug_image(sharp_binary_image, "10_sharp_binary")
    
    # --- NEW Step 4: Final Edge-Preserving Smoothing ---
    print("Applying mild, edge-preserving bilateral filter...")
    # This filter smooths while keeping text edges sharp.
    # d: Diameter of the pixel neighborhood.
    # sigmaColor/sigmaSpace: Higher values give a more pronounced but "smarter" blur.
    final_output_image = cv2.bilateralFilter(sharp_binary_image, d=5, sigmaColor=75, sigmaSpace=75)
    save_debug_image(final_output_image, "11_final_bilateral_smoothed")

    # --- Step 5: Save Final Result ---
    try:
        cv2.imwrite(output_path, final_output_image)
        print(f"Successfully saved final processed image to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving final image: {e}")
        return False

# --- Main execution block ---
if __name__ == "__main__":
    input_filename = "D:\Anirudh\mini_project_final\mini_project_data\images\sample3.jpg"
    if not os.path.exists(input_filename):
       print(f"FATAL ERROR: The input file '{input_filename}' was not found.")
    else:
        output_filename = "sample3_output_v2.jpg"
        debug_folder = "debug_images_sample3_v2"

        # --- Parameters for Scanner Pro Method ---
        BLOCK_SIZE = 51
        C_CONSTANT = 8
        # --- End Parameters ---

        success = preprocess_document_v27_bilateral_smooth(
            input_filename,
            output_filename,
            debug_folder,
            apply_post_warp_deskew=True,
            adaptive_thresh_blocksize=BLOCK_SIZE,
            adaptive_thresh_C=C_CONSTANT
        )
        if success:
            print("Processing finished successfully.")
        else:
            print("Processing failed.")