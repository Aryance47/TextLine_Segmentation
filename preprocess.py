import os
import cv2
import numpy as np
import xmltodict

def pair_files(image_dir):
    """Pair JPG images with their XML annotations."""
    pairs = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.jpg'):
            xml_file = os.path.join(image_dir, 'page', img_file.replace('.jpg', '.xml'))
            if os.path.exists(xml_file):
                pairs.append((os.path.join(image_dir, img_file), xml_file))
    return pairs

def parse_xml(xml_path):
    """Extract text line coordinates from XML."""
    with open(xml_path, 'r') as f:
        xml_data = xmltodict.parse(f.read())
    
    coords = []
    
    # Case 1: Standard PAGE XML format
    if 'PcGts' in xml_data and 'Page' in xml_data['PcGts']:
        page = xml_data['PcGts']['Page']
        if 'TextRegion' in page:
            regions = page['TextRegion'] if isinstance(page['TextRegion'], list) else [page['TextRegion']]
            for region in regions:
                if 'TextLine' in region:
                    lines = region['TextLine'] if isinstance(region['TextLine'], list) else [region['TextLine']]
                    for line in lines:
                        if 'Coords' in line and '@points' in line['Coords']:
                            points = line['Coords']['@points'].split()
                            coords.append([tuple(map(int, p.split(','))) for p in points])
    
    # # Case 2: Alternative XML structure (adjust as needed)
    # elif 'AnotherRootTag' in xml_data:  # Replace with actual root tag if different
    #     # Add parsing logic for your specific format here
    #     pass
    
    return coords

def create_mask(coords, output_size=(1024, 1024)):
    """Convert coordinates to a binary mask."""
    mask = np.zeros(output_size, dtype=np.uint8)
    for polygon in coords:
        cv2.fillPoly(mask, [np.array(polygon)], 255)
    return mask

def preprocess_data(image_dir, output_dir):
    """Save preprocessed images/masks as NPY files."""
    os.makedirs(output_dir, exist_ok=True)
    pairs = pair_files(image_dir)
    for img_path, xml_path in pairs:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1024, 1024))
        coords = parse_xml(xml_path)
        mask = create_mask(coords)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        np.save(f"{output_dir}/{base_name}_img.npy", img)
        np.save(f"{output_dir}/{base_name}_mask.npy", mask)

if __name__ == "__main__":
    preprocess_data(
        image_dir=r"C:\Users\91992\READ-ICDAR2017-cBAD-dataset-v4\Train-Baseline Competition - Complex Documents\BHIC_Akten",
        output_dir="data/preprocessed/train/"
    )