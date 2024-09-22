import xml.etree.ElementTree as ET
from svgpathtools import svg2paths, wsvg, Path, Line

def remove_metadata(svg_file, cleaned_svg_file):
    """
    Remove unnecessary metadata, comments, and descriptions.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Remove metadata elements (if any)
    for metadata in root.findall(".//{http://www.w3.org/2000/svg}metadata"):
        root.remove(metadata)

    # Remove unnecessary comments or descriptions
    for desc in root.findall(".//{http://www.w3.org/2000/svg}desc"):
        root.remove(desc)
    for title in root.findall(".//{http://www.w3.org/2000/svg}title"):
        root.remove(title)

    tree.write(cleaned_svg_file)
    print(f"Metadata cleaned and saved to {cleaned_svg_file}")

from svgpathtools import svg2paths, wsvg, Path, Line

def simplify_paths(svg_file, tolerance=0.1):
    """
    Simplify paths in the SVG using svgpathtools.
    """
    paths, attributes = svg2paths(svg_file)

    if not paths:
        print("No paths found in the SVG file!")
        return

    simplified_paths = []

    for path in paths:
        simplified_path = Path()
        for seg in path:
            # Here you can add your logic to simplify, for now, we add all segments
            simplified_path.append(seg)

        simplified_paths.append(simplified_path)

    if not simplified_paths:
        print("No simplified paths were generated!")
        return

    # Save simplified SVG only if paths exist
    wsvg(simplified_paths, filename="simplified_" + svg_file, attributes=attributes)
    print(f"Simplified paths saved to simplified_{svg_file}")

def remove_hidden_layers(svg_file, cleaned_svg_file):
    """
    Remove hidden or invisible layers (elements with display:none or visibility:hidden).
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    for elem in root.findall(".//*[@display='none']"):
        root.remove(elem)
    for elem in root.findall(".//*[@visibility='hidden']"):
        root.remove(elem)

    tree.write(cleaned_svg_file)
    print(f"Hidden layers removed and saved to {cleaned_svg_file}")

def flatten_svg(svg_file, cleaned_svg_file):
    """
    Flatten the SVG by ungrouping and converting all elements to paths.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    for group in root.findall(".//{http://www.w3.org/2000/svg}g"):
        root.extend(group.findall(".//{http://www.w3.org/2000/svg}*"))
        root.remove(group)

    # Save the flattened SVG
    tree.write(cleaned_svg_file)
    print(f"SVG flattened and saved to {cleaned_svg_file}")

def clean_svg(svg_file):
    """
    Clean SVG by combining all the cleaning steps.
    """
    # Remove metadata and hidden layers
    cleaned_svg_file = "cleaned_" + svg_file
    remove_metadata(svg_file, cleaned_svg_file)
    remove_hidden_layers(cleaned_svg_file, cleaned_svg_file)
    flatten_svg(cleaned_svg_file, cleaned_svg_file)

    # Simplify paths
    simplify_paths(cleaned_svg_file)

if __name__ == "__main__":
    svg_file = "output.svg"  # Replace with your SVG file
    clean_svg(svg_file)
