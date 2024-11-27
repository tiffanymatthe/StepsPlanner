from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the size and color
size = (512, 512)
color = (255,255,255)  # Red color in RGB

# Create the image
image = Image.new("RGB", size, color)

# Save the image
image.save(f"{script_dir}/solid_color.png")
