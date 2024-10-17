import xml.etree.ElementTree as ET
import os

# Function to update arm size in the Mujoco XML
def modify_mujoco_sizes(xml_path, arm_scale_factor, leg_scale_factor, save_dir):
    base, ext = os.path.splitext(xml_path)
    folder = f"{os.path.dirname(save_dir)}/temp_xml"
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = f"{folder}/walker3d_modified_{arm_scale_factor}_{leg_scale_factor}{ext}"

    if os.path.isfile(output_path):
        return output_path

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find all geometries for the arms and modify their size
    for geom in root.iter('geom'):
        name = geom.attrib.get('name')
        
        # Check if this geometry is part of the arm (right and left upper or lower arms)
        if name and ('uarm' in name or 'larm' in name):  # Match arms by name (right_uarm, left_larm, etc.)
            # Split current 'size' value (since it might include multiple values, like length and radius)
            sizes = geom.attrib.get('size').split()
            
            # Modify the radius (first value), keeping the original length
            sizes[0] = str(float(sizes[0]) * arm_scale_factor)
            
            # Update the 'size' attribute with the new size for the arm thickness
            geom.attrib['size'] = ' '.join(sizes)
        elif name and name in ["right_thigh1", "right_shin1", "left_thigh1", "left_shin1"]:
            sizes = geom.attrib.get('size').split()
            sizes[0] = str(float(sizes[0]) * leg_scale_factor)
            geom.attrib['size'] = ' '.join(sizes)

    # Write the modified XML to a new file
    tree.write(output_path)

    return output_path
