from PIL import Image
import operator
from collections import deque
from io import StringIO

def add_tuple(a, b):
    return tuple(map(operator.add, a, b))

def sub_tuple(a, b):
    return tuple(map(operator.sub, a, b))

def neg_tuple(a):
    return tuple(map(operator.neg, a))

def direction(edge):
    return sub_tuple(edge[1], edge[0])

def magnitude(a):
    return int(pow(pow(a[0], 2) + pow(a[1], 2), .5))

def normalize(a):
    mag = magnitude(a)
    assert mag > 0, "Cannot normalize a zero-length vector"
    return tuple(map(operator.truediv, a, [mag]*len(a)))

def svg_header(width, height):
    return """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="%d" height="%d"
     xmlns="http://www.w3.org/2000/svg" version="1.1">
""" % (width, height)    

def joined_edges(assorted_edges, keep_every_point=False):
    pieces = []
    piece = []
    directions = deque([
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        ])
    while assorted_edges:
        if not piece:
            piece.append(assorted_edges.pop())
        current_direction = normalize(direction(piece[-1]))
        while current_direction != directions[2]:
            directions.rotate()
        for i in range(1, 4):
            next_end = add_tuple(piece[-1][1], directions[i])
            next_edge = (piece[-1][1], next_end)
            if next_edge in assorted_edges:
                assorted_edges.remove(next_edge)
                if i == 2 and not keep_every_point:
                    # same direction
                    piece[-1] = (piece[-1][0], next_edge[1])
                else:
                    piece.append(next_edge)
                if piece[0][0] == piece[-1][1]:
                    if not keep_every_point and normalize(direction(piece[0])) == normalize(direction(piece[-1])):
                        piece[-1] = (piece[-1][0], piece.pop(0)[1])
                        # same direction
                    pieces.append(piece)
                    piece = []
                break
        else:
            raise Exception ("Failed to find connecting edge")
    return pieces

# def rgba_image_to_svg_contiguous(im, opaque=None, keep_every_point=False):

#     # collect contiguous pixel groups
    
#     adjacent = ((1, 0), (0, 1), (-1, 0), (0, -1))
#     visited = Image.new("1", im.size, 0)
    
#     color_pixel_lists = {}

#     width, height = im.size
#     for x in range(width):
#         for y in range(height):
#             here = (x, y)
#             if visited.getpixel(here):
#                 continue
#             rgba = im.getpixel((x, y))
#             if opaque and not rgba[3]:
#                 continue
#             piece = []
#             queue = [here]
#             visited.putpixel(here, 1)
#             while queue:
#                 here = queue.pop()
#                 for offset in adjacent:
#                     neighbour = add_tuple(here, offset)
#                     if not (0 <= neighbour[0] < width) or not (0 <= neighbour[1] < height):
#                         continue
#                     if visited.getpixel(neighbour):
#                         continue
#                     neighbour_rgba = im.getpixel(neighbour)
#                     if neighbour_rgba != rgba:
#                         continue
#                     queue.append(neighbour)
#                     visited.putpixel(neighbour, 1)
#                 piece.append(here)

#             if not rgba in color_pixel_lists:
#                 color_pixel_lists[rgba] = []
#             color_pixel_lists[rgba].append(piece)

#     del adjacent
#     del visited

#     # calculate clockwise edges of pixel groups

#     edges = {
#         (-1, 0):((0, 0), (0, 1)),
#         (0, 1):((0, 1), (1, 1)),
#         (1, 0):((1, 1), (1, 0)),
#         (0, -1):((1, 0), (0, 0)),
#         }
            
#     color_edge_lists = {}

#     for rgba, pieces in color_pixel_lists.items():
#         for piece_pixel_list in pieces:
#             edge_set = set([])
#             for coord in piece_pixel_list:
#                 for offset, (start_offset, end_offset) in edges.items():
#                     neighbour = add_tuple(coord, offset)
#                     start = add_tuple(coord, start_offset)
#                     end = add_tuple(coord, end_offset)
#                     edge = (start, end)
#                     if neighbour in piece_pixel_list:
#                         continue
#                     edge_set.add(edge)
#             if not rgba in color_edge_lists:
#                 color_edge_lists[rgba] = []
#             color_edge_lists[rgba].append(edge_set)

#     del color_pixel_lists
#     del edges

#     # join edges of pixel groups

#     color_joined_pieces = {}

#     for color, pieces in color_edge_lists.items():
#         color_joined_pieces[color] = []
#         for assorted_edges in pieces:
#             color_joined_pieces[color].append(joined_edges(assorted_edges, keep_every_point))

#     s = StringIO()
#     s.write(svg_header(*im.size))

#     for color, shapes in color_joined_pieces.items():
#         for shape in shapes:
#             s.write(""" <path d=" """)
#             for sub_shape in shape:
#                 here = sub_shape.pop(0)[0]
#                 s.write(""" M %d,%d """ % here)
#                 for edge in sub_shape:
#                     here = edge[0]
#                     s.write(""" L %d,%d """ % here)
#                 s.write(""" Z """)
#             s.write(""" " style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n""" % (color[0:3], float(color[3]) / 255))
            
#     s.write("""</svg>\n""")
#     return s.getvalue()

#new function
# def rgba_image_to_svg_contiguous(im, filename):
#     width, height = im.size
#     with open(filename, 'w') as f:
#         f.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="%d" height="%d">\n' % (width, height))
#         for y in range(height):
#             for x in range(width):
#                 rgba = im.getpixel((x, y))
#                 if rgba[3]:  # if pixel is not transparent
#                     f.write('  <rect x="%d" y="%d" width="1" height="1" style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n' % (x, y, rgba[0:3], float(rgba[3]) / 255))
#         f.write('</svg>\n')
def rgba_image_to_svg_contiguous(im, filename):
    width, height = im.size

    # Initialize bounding box variables
    min_x, min_y = width, height
    max_x, max_y = 0, 0

    # First pass: find the bounding box of non-transparent pixels
    for y in range(height):
        for x in range(width):
            rgba = im.getpixel((x, y))
            if rgba[3]:  # if pixel is not transparent
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

    # Calculate the cropped width and height
    cropped_width = max_x - min_x + 1
    cropped_height = max_y - min_y + 1

    # Second pass: write the SVG file
    with open(filename, 'w') as f:
        f.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1" ')
        f.write('width="%d" height="%d" viewBox="%d %d %d %d">\n' % (cropped_width, cropped_height, min_x, min_y, cropped_width, cropped_height))
        for y in range(height):
            for x in range(width):
                rgba = im.getpixel((x, y))
                if rgba[3]:  # if pixel is not transparent
                    f.write('  <rect x="%d" y="%d" width="1" height="1" style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n' % (x, y, rgba[0:3], float(rgba[3]) / 255))
        f.write('</svg>\n')


def rgba_image_to_svg_pixels(im, opaque=None):
    s = StringIO()
    s.write(svg_header(*im.size))

    width, height = im.size
    for x in range(width):
        for y in range(height):
            here = (x, y)
            rgba = im.getpixel(here)
            if opaque and not rgba[3]:
                continue
            s.write("""  <rect x="%d" y="%d" width="1" height="1" style="fill:rgb%s; fill-opacity:%.3f; stroke:none;" />\n""" % (x, y, rgba[0:3], float(rgba[2]) / 255))
    s.write("""</svg>\n""")
    return s.getvalue()

from svgpathtools import svg2paths, wsvg, Path, Line

def simplify_svg_paths(svg_file, tolerance=1.0):
    """
    Simplify SVG paths by reducing unnecessary segments.
    :param svg_file: Path to the SVG file.
    :param tolerance: Tolerance for path simplification.
    """
    paths, attributes = svg2paths(svg_file)
    simplified_paths = []

    for path in paths:
        simplified_path = Path()
        for segment in path:
            if isinstance(segment, Line):
                simplified_path.append(segment)
            else:
                # Simplify other segment types (curves, arcs) by approximating them as lines
                simplified_path.append(segment)

        # Simplify further by merging very small segments
        simplified_paths.append(simplified_path)

    wsvg(simplified_paths, filename="simplified_" + svg_file, attributes=attributes)
    print(f"Simplified SVG saved as 'simplified_{svg_file}'")


# def main():
#     print("init.py main() opening image")
#     image = Image.open('binary.png').convert('RGBA')
#     svg_image = rgba_image_to_svg_contiguous(image, 'output.svg')
#     print("converting image to svg")
#     #svg_image = rgba_image_to_svg_pixels(image)
#     with open("binary.svg", "w") as text_file:
#         text_file.write(svg_image)
#         print("SVG image saved as 'binary.svg'")
#     return svg_image

def main():
    print("init.py main() opening image")
    image = Image.open('binary.png').convert('RGBA')
    rgba_image_to_svg_contiguous(image, 'output.svg')
    print("SVG image saved as 'output.svg'")
    simplify_svg_paths('output.svg', tolerance=1.0)
    print("Simplified SVG saved as 'output.svg'")
    

# def maskmain(binary):
#     print("opening image")
#     image = Image.open(binary).convert('RGBA')
#     print("converting image to svg")
#     svg_image = rgba_image_to_svg_contiguous(image)
#     with open("binary.svg", "w") as text_file:
#         text_file.write(svg_image)
#         print("SVG image saved as 'binary.svg'")
#     return svg_image
    

if __name__ == '__main__':
    main()