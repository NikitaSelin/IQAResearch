import cv2
import os
from tqdm import tqdm


def perform_bicubic_on_folder(folder_path: str, output_path: str):
    image_files = os.listdir(folder_path)
    if not os.path.exists:
        os.mkdir(output_path)
    for image_file in tqdm(image_files):
        image = cv2.imread(f"{folder_path}/{image_file}")
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{output_path}/{image_file}", image)


if __name__ == "__main__":
    perform_bicubic_on_folder("downscaled_images", "bicubic")
