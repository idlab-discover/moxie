"""Check the real SVG written by the native display contract test."""
from pathlib import Path
import xml.etree.ElementTree as ET

path = Path('builds/calibration-display.svg')
root = ET.parse(path).getroot()
ns = {'svg': 'http://www.w3.org/2000/svg'}
assert root.attrib['viewBox'] == '0 0 720 620'
assert len(root.findall('.//svg:path[@data-reference="true"]', ns)) == 1
curves = root.findall('.//svg:polyline', ns)
assert len(curves) == 2
assert curves[1].attrib['stroke'] == '#d97706'
assert curves[1].attrib['stroke-dasharray'] == '7 5'
for curve in curves:
    points = [tuple(map(float, point.split(','))) for point in curve.attrib['points'].split()]
    assert len(points) == 3
    assert all(100 <= x <= 640 and 80 <= y <= 520 for x, y in points)
labels = [text.text for text in root.findall('.//svg:text', ns)]
assert 'Before <fit>' in labels
assert 'After & fit' in labels
assert 'Mean predicted probability (Positive class: 1)' in labels
print('PASS: native SVG structure, coordinates, legend and escaping')
