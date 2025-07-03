import cv2
import numpy as np
import os
import math
from deskew import determine_skew

# --- order_points, perspective_transform, segment_intersection
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

def preprocess_document_hough_v15(input_path, output_path, debug_folder="debug_images_v15",
                                   apply_post_warp_deskew=True,
                                   median_blur_kernel=5,        # Kernel size for median blur (must be odd, 0 or 1 to disable)
                                   adaptive_thresh_blocksize=25,
                                   adaptive_thresh_C=10):
    """
    Hough perspective correction with median blur before tunable binarization (v15).
    """
    print(f"--- Starting Preprocessing v15 (Median Blur + Tune Binarize) for: {input_path} ---")

    # --- Setup Debug Folder & Save Utility ---
    if not os.path.exists(debug_folder): os.makedirs(debug_folder)
    def save_debug_image(img_data, filename_base, draw_on=None):
        if img_data is not None and img_data.size > 0:
            display_img = draw_on if draw_on is not None else img_data
            fname = os.path.join(debug_folder, f"{filename_base}.png")
            try: cv2.imwrite(fname, display_img); print(f"Saved Debug Image: {fname}")
            except Exception as e: print(f"Error saving debug {fname}: {e}")
        else: print(f"Skipping save for debug {filename_base} (Empty)")


    # --- Steps 1-5: Load, Preprocess, Hough, Filter, Find Edges 
    img_orig = cv2.imread(input_path);
    if img_orig is None: return False
    print("Image loaded."); img_debug_lines=img_orig.copy(); h_orig,w_orig=img_orig.shape[:2]
    save_debug_image(img_orig, "01_original")
    gray=cv2.cvtColor(img_orig,cv2.COLOR_BGR2GRAY); blurred=cv2.GaussianBlur(gray,(5,5),0)
    edged=cv2.Canny(blurred,50,150); print("Applied Gray, Blur, Canny."); save_debug_image(edged, "02_edged")
    lines=cv2.HoughLinesP(edged,1,np.pi/180,threshold=100,minLineLength=50,maxLineGap=15)
    if lines is None: print("Error: No Hough lines."); cv2.imwrite(output_path, gray); return False
    print(f"Detected {len(lines)} lines.")
    # ...(line filtering logic same)...
    horizontal_lines = []; vertical_lines = []
    angle_tolerance_deg=15; min_len=0.1*min(w_orig,h_orig)
    for line in lines:
        x1,y1,x2,y2=line[0]; length=math.sqrt((x2-x1)**2+(y2-y1)**2)
        if length<min_len: continue
        angle_deg=90 if x2-x1==0 else math.degrees(math.atan(abs(y2-y1)/abs(x2-x1)))
        if angle_deg<angle_tolerance_deg: horizontal_lines.append(line[0])
        elif abs(angle_deg-90)<angle_tolerance_deg: vertical_lines.append(line[0])
    print(f"Filtered to {len(horizontal_lines)} H, {len(vertical_lines)} V lines.")
    if not horizontal_lines or not vertical_lines: print("Error: Not enough H/V lines."); cv2.imwrite(output_path, gray); return False
    top_line=min(horizontal_lines,key=lambda l:(l[1]+l[3])/2); bottom_line=max(horizontal_lines,key=lambda l:(l[1]+l[3])/2)
    left_line=min(vertical_lines,key=lambda l:(l[0]+l[2])/2); right_line=max(vertical_lines,key=lambda l:(l[0]+l[2])/2)
    # ...(corner finding logic same)...
    corners = []
    tl = segment_intersection(top_line, left_line)
    if tl:
        corners.append(tl)
    
    tr = segment_intersection(top_line, right_line)
    if tr:
        corners.append(tr)
    
    br = segment_intersection(bottom_line, right_line)
    if br:
        corners.append(br)
    
    bl = segment_intersection(bottom_line, left_line)
    if bl:
        corners.append(bl)
    print(f"Found {len(corners)} corners.")
    # --- End of copied logic ---

    # 6. Order Corners & Warp
    final_output_image = gray
    success_process = False
    warped_gray = None

    if len(corners) == 4:
        print("Applying perspective transform...")
        corner_array = np.array(corners, dtype="float32")
        warped_color = perspective_transform(img_orig, corner_array)
        if warped_color is not None:
            print("Perspective transform successful.")
            save_debug_image(warped_color, "07_warped_color")
            warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
            save_debug_image(warped_gray, "08_warped_gray")
            success_process = True
        else: print("Error during perspective transform."); warped_gray = gray
    else: print(f"Warning: Did not find 4 corners."); warped_gray = gray

    # 7. Optional Post-Warp Rotational Deskew
    image_to_filter = warped_gray
    if success_process and apply_post_warp_deskew and warped_gray is not None:
        print("Applying post-warp rotational deskew...")
        try:
            angle = determine_skew(warped_gray)
            if angle is None or abs(angle)<0.1 or abs(angle)>45: print(f"Skipping deskew (angle={angle}).")
            else:
                print(f"Deskew angle: {angle:.2f}"); (h_w,w_w)=warped_gray.shape[:2]; center_w=(w_w//2,h_w//2)
                M_w=cv2.getRotationMatrix2D(center_w,angle,1.0)
                image_to_filter = cv2.warpAffine(warped_gray,M_w,(w_w,h_w),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=(255))
                save_debug_image(image_to_filter, "08a_post_warp_deskewed")
        except Exception as e: print(f"Error deskewing: {e}.")
    # --- image_to_filter now holds the warped, optionally deskewed grayscale image ---

    # 8. Median Blurring (to remove scratches/noise *before* binarization)
    image_to_binarize = image_to_filter # Default if blurring skipped
    if median_blur_kernel > 1 and image_to_filter is not None and image_to_filter.size > 0 :
         # Kernel size must be odd
         ksize = median_blur_kernel if median_blur_kernel % 2 == 1 else median_blur_kernel + 1
         print(f"Applying Median Blur (kernel={ksize})...")
         image_to_binarize = cv2.medianBlur(image_to_filter, ksize)
         save_debug_image(image_to_binarize, "08b_median_blurred")
    else:
        print("Skipping Median Blur.")


    # 9. Final Tunable Binarization
    if image_to_binarize is not None and image_to_binarize.size > 0:
        print(f"Applying final adaptive thresholding (blockSize={adaptive_thresh_blocksize}, C={adaptive_thresh_C})...")
        final_output_image = cv2.adaptiveThreshold(image_to_binarize, 255,
                                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY,
                                                   adaptive_thresh_blocksize,
                                                   adaptive_thresh_C)
        print("Binarization complete.")
        save_debug_image(final_output_image, "09_final_binary")
    else:
        print("Error: Image data for binarization is empty."); final_output_image = gray
        save_debug_image(gray, "09_fallback_gray_final")

    # 10. Save Final Result
    try:
        if final_output_image is not None and final_output_image.size > 0:
            cv2.imwrite(output_path, final_output_image)
            print(f"Successfully saved final processed image to: {output_path}")
            return True
        else: return False
    except Exception as e: print(f"Error saving final image: {e}"); return False

# --- Main execution block ---
if __name__ == "__main__":
    input_filename = "D:\Anirudh\mini_project_final\mini_project_data\images\sample3.jpg"
    output_filename = "sample3_output.jpg"
    debug_folder = "debug_images_sample3"

    # --- Parameters to Tune ---
    POST_WARP_DESKEW = True  # Keep optional deskew
    MEDIAN_KERNEL = 1        # Median blur kernel (e.g., 3 or 5). 0 or 1 disables. MUST BE ODD.
    BLOCK_SIZE = 25          # Adaptive threshold block size (odd)
    C_CONSTANT = 10          # Adaptive threshold constant
    # --- End Parameters ---

    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
    else:
        success = preprocess_document_hough_v15(
            input_filename,
            output_filename,
            debug_folder,
            apply_post_warp_deskew=POST_WARP_DESKEW,
            median_blur_kernel=MEDIAN_KERNEL,
            adaptive_thresh_blocksize=BLOCK_SIZE,
            adaptive_thresh_C=C_CONSTANT
        )
        if success: print("Processing finished.")
        else: print("Processing failed.")