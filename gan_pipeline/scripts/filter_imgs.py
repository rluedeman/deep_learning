import os

from absl import app, flags
from img2vec_pytorch import Img2Vec
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import sys
# TODO: Not sure why system PYTHONPATH doesn't work here.
sys.path.append('C:\\Users\\rober\\Dropbox\\projects\\fun\\deep_learning')

from gan_pipeline.utils import validate_images


FLAGS = flags.FLAGS
flags.DEFINE_string("image_dir", None, "directory of images + targets")
flags.mark_flag_as_required('image_dir')


def filter_images(image_dir):
    """
    Given an iamge_dir, expects to find a "Raw" and  "Target" subdirectory.

    Will filter the images in the "Raw" directory to only include images that
    are near the target image in the "Target" directory.
    """
    # First, validate the images are in the right subdirectories and all of the same resolution
    raw_dir = os.path.join(image_dir, "Raw")
    target_dir = os.path.join(image_dir, "Target")
    validate_images(raw_dir, target_dir)

    # Compute the similarity between the raw and target images
    compute_similarity(raw_dir, target_dir)


def compute_similarity(raw_dir, target_dir):
    img2vec = Img2Vec(cuda=True, model="resnet18", layer="default")
    
    # Get the vectorized representations of the target images
    TARGET_IMG_INDEX = 8
    target_image_paths = [os.path.join(target_dir, f) for f in os.listdir(target_dir)][:10]
    target_vectors = []
    for target_image_path in tqdm(target_image_paths, desc="Computing Target Vectors"):
        target_image = Image.open(target_image_path)
        target_vectors.append(img2vec.get_vec(target_image))

    # Get the vectorized representations of the raw images
    raw_image_paths = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir)][:2000]
    closest = 0
    closest_path = None
    for raw_image_path in tqdm(raw_image_paths, desc="Computing Raw Vectors"):
        raw_vec = img2vec.get_vec(Image.open(raw_image_path))
        for i in range(10):
            similarity = cosine_similarity(raw_vec.reshape((1, -1)),
                                           target_vectors[TARGET_IMG_INDEX].reshape((1, -1)))[0][0]
        if similarity > closest:
            closest = similarity
            closest_path = raw_image_path
    
    print(f"Closest image is {closest_path} with similarity {closest}")
    target_image = Image.open(target_image_paths[TARGET_IMG_INDEX])
    closest_image = Image.open(closest_path)
    plt.subplot(1, 2, 1)
    plt.imshow(target_image)
    plt.subplot(1, 2, 2)
    plt.imshow(closest_image)
    plt.show()

    



def main(argv):
    """
    Main function. Passes the args to the filter_images function.
    """
    filter_images(FLAGS.image_dir)
    


if __name__ == '__main__':
    app.run(main)