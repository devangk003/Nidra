import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to add indentation to ElementTree
def indent_xml(elem, level=0, indent="  "):
    """Add proper indentation to XML elements for pretty printing."""
    i = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level + 1, indent)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def convert_rml_file(file_path):
    """Convert RML file from nested format to simplified format with proper indentation."""
    try:
        # Parse the original XML
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Find all Stage elements in the nested structure
        stage_elements = root.findall('.//Stage')
        
        if not stage_elements:
            logging.warning(f"No Stage elements found in {file_path}")
            return False
            
        # Create new root element
        new_root = ET.Element('SleepStages')
        
        # Copy each Stage element to the new root, formatting Start as float
        for stage in stage_elements:
            new_stage = ET.SubElement(new_root, 'Stage')
            
            # Get attributes and convert Start to float format
            stage_type = stage.get('Type')
            start_time = stage.get('Start')
            
            if not stage_type or not start_time:
                continue  # Skip stages with missing attributes
                
            try:
                start_float = f"{float(start_time):.1f}"
            except ValueError:
                # If conversion fails, use original value
                start_float = start_time
            
            # Set attributes in new order (Start first, then Type)
            new_stage.set('Start', start_float)
            new_stage.set('Type', stage_type)
        
        # Add proper indentation to the XML
        indent_xml(new_root)
        
        # Create new ElementTree
        new_tree = ET.ElementTree(new_root)
        
        # Write to file with XML declaration
        with open(file_path, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            new_tree.write(f, encoding='utf-8')
        
        return True
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False

def main():
    """Process all RML files in the dataset."""
    # Check if a directory path is provided as command-line argument
    if len(sys.argv) > 1:
        rml_dir = sys.argv[1]
    else:
        rml_dir = os.path.join('dataset', 'APNEA_RML')
    
    # Verify directory exists
    if not os.path.exists(rml_dir):
        logging.error(f"Directory not found: {rml_dir}")
        return
    
    success_count = 0
    error_count = 0
    
    # Walk through all directories
    for root, dirs, files in os.walk(rml_dir):
        rml_files = [f for f in files if f.endswith('.rml')]
        if not rml_files:
            continue
            
        for file in tqdm(rml_files, desc=f"Processing files in {root}"):
            file_path = os.path.join(root, file)
            logging.info(f"Converting {file_path}")
            
            if convert_rml_file(file_path):
                success_count += 1
            else:
                error_count += 1
    
    logging.info(f"Conversion complete. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()