import os
from PIL import Image
import numpy as np


INPUT_DIR = "./output2" 


OUTPUT_DIR = "./output3" 


ALPHA_THRESHOLD = 10



def align_transparent_images():
   
    if not os.path.exists(INPUT_DIR):
        
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
       

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.png')]
    if not image_files:
        
        return

    bottom_coords = {}
    max_bottom_y = 0
    image_size = None
    
  
    for filename in image_files:
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with Image.open(filepath) as img:
               
                img_rgba = img.convert('RGBA')
                if image_size is None:
                    image_size = img_rgba.size

                img_array = np.array(img_rgba)
                
                alpha_channel = img_array[:, :, 3]
                
                non_transparent_coords = np.argwhere(alpha_channel > ALPHA_THRESHOLD)
                
                if non_transparent_coords.size > 0:
               
                    current_bottom_y = non_transparent_coords[:, 0].max()
                    bottom_coords[filename] = current_bottom_y
                 
                    if current_bottom_y > max_bottom_y:
                        max_bottom_y = current_bottom_y
                    print(f" File: {filename}, Model bottom Y coordinate: {current_bottom_y}")
                else:
                    bottom_coords[filename] = -1 
                 
        except Exception as e:
            print(f"fail: {e}")


    if max_bottom_y == 0:
      
        return

    print(f"All models will be aligned to the baseline of the Y coordinate: {max_bottom_y}.")

    for filename in image_files:
        if bottom_coords.get(filename, -1) == -1:
            continue

        try:
            with Image.open(os.path.join(INPUT_DIR, filename)).convert('RGBA') as img:
             
                shift_y = max_bottom_y - bottom_coords[filename]
         
                new_img = Image.new('RGBA', image_size, (0, 0, 0, 0))
                
                new_img.paste(img, (0, shift_y))
                
                output_path = os.path.join(OUTPUT_DIR, filename)
                new_img.save(output_path)
                print(f"  Aligned image generated: {filename} (shifted down by {shift_y} pixels)")
        except Exception as e:
            print(f"fail: {e}")

    print("Processing completed!")

if __name__ == "__main__":
    align_transparent_images()