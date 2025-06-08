import pytesseract
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image

def generate_svg_from_detected_elements(image_path, detected_elements, output_svg_path="output.svg"):
    """
    Generate an SVG file from detected layout elements of a document.

    Args:
        image_path (str): Path to the original image.
        detected_elements (dict): Dictionary with keys 'logos', 'text', 'tables' and values as lists of (x, y, w, h).
        output_svg_path (str): File path to save the generated SVG.
    """
    # Load the original image
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Create base SVG element
    svg = Element('svg', width=str(img_w), height=str(img_h), xmlns="http://www.w3.org/2000/svg")

    def add_box_and_text(box, color, label, font_size=16):
        x, y, w, h = box
        # Draw rectangle
        SubElement(svg, "rect", x=str(x), y=str(y), width=str(w), height=str(h),
                   fill="none", stroke=color, stroke_width="1")
        # Crop the image region
        roi = img.crop((x, y, x + w, y + h))
        # Extract text with OCR
        text = pytesseract.image_to_string(roi, config="--psm 6").strip().replace('\n', ' ')
        if text:
            SubElement(svg, "text", x=str(x + 2), y=str(y + h - 5),
                       font_family="Times New Roman", font_size=str(font_size),
                       fill="black").text = text

    # Add all elements to SVG
    for box in detected_elements.get("logos", []):
        add_box_and_text(box, "red", "Logo", font_size=14)

    for box in detected_elements.get("text", []):
        add_box_and_text(box, "green", "Text", font_size=18)

    for box in detected_elements.get("tables", []):
        add_box_and_text(box, "blue", "Table", font_size=16)

    # Write SVG to file
    svg_string = tostring(svg, 'utf-8')
    pretty_svg = parseString(svg_string).toprettyxml(indent="  ")

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(pretty_svg)

    return output_svg_path
