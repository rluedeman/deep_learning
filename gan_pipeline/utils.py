from itertools import chain
import os

from PIL import Image
from tqdm import tqdm




def validate_images(raw_dir, target_dir):
    """
    Validate that both raw_dir and target_dir exist, contain images, and that the images are all of
    the same resolution.
    """
    # Directories exist?
    assert os.path.exists(raw_dir), "Raw subdirectory does not exist"
    assert os.path.exists(target_dir), "Target subdirectory does not exist"

    # Directories contain images?
    assert len(os.listdir(raw_dir)) > 0, "Raw subdirectory is empty"
    assert len(os.listdir(target_dir)) > 0, "Target subdirectory is empty"

    # Images are the same resolution?
    raw_image_paths = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)]
    target_image_paths = [os.path.join(target_dir, f) for f in os.listdir(target_dir)]
    all_image_paths = list(chain(raw_image_paths, target_image_paths))
    first_image = Image.open(all_image_paths[0])
    target_img_size = first_image.size
    for img_path in tqdm(all_image_paths, desc=f"Confirming Img Shape of {target_img_size}"):
        try:
            img = Image.open(img_path)
        except:
            raise Exception(f"Could not open image: {img_path}")
        assert img.size == target_img_size, f"Image {img_path} is not the same size as the first image"        

    print("Successfully validated %d raw and %d target images." % (len(raw_image_paths), len(target_image_paths)))
