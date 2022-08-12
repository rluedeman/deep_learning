import os

from absl import app, flags

import sys
# TODO: Not sure why system PYTHONPATH doesn't work here.
sys.path.append('C:\\Users\\rober\\Dropbox\\projects\\fun\\deep_learning')
from gan_pipeline.utils import validate_images
from gan_pipeline.similarity import SimilarImgGetter


FLAGS = flags.FLAGS
flags.DEFINE_string("image_dir", None, "directory of images + targets")
flags.mark_flag_as_required('image_dir')


def calibrate_similarity(image_dir):
    """
    Given an iamge_dir, expects to find a "Raw" and  "Target" subdirectory.

    Will provide visualizations that help choose the best threshold for the similarity metric.
    """
    # First, validate the images are in the right subdirectories and all of the same resolution
    raw_dir = os.path.join(image_dir, "Raw")
    target_dir = os.path.join(image_dir, "Target")
    validate_images(raw_dir, target_dir)

    # Compute the similarity between the raw and target images
    sim = SimilarImgGetter(target_img_dir=target_dir, raw_img_dir=raw_dir)
    # targ, raw = sim.get_most_similar_pair()
    sim.plot_similarity_disttribution()
    sim.plot_images_in_similarity_range(0, 1)



def main(argv):
    """
    Main function. Passes the args to the filter_images function.
    """
    calibrate_similarity(FLAGS.image_dir)
    

if __name__ == '__main__':
    app.run(main)