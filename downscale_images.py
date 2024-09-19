import os
import cv2
from tqdm import tqdm


def downscale_images(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = os.listdir(folder_path)
    
    for file in tqdm(files):
        input_path = os.path.join(folder_path, file)
        
        img = cv2.imread(input_path)
        
        if img is not None:
            height, width = img.shape[:2]
            
            new_width = int(width / 4)
            new_height = int(height / 4)
            
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            output_path = os.path.join(output_folder, file)
            
            cv2.imwrite(output_path, resized_img)
        else:
            print(f"Could not read {input_path}")

# Example usage:
if __name__ == "__main__":
    input_folder = "original_images"
    output_folder = "downscaled_images"
    
    downscale_images(input_folder, output_folder)
