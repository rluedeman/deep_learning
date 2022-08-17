import os
import random
import sys

from img2vec_pytorch import Img2Vec
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm


# TODO: Refactor to not require raw images.
class SimilarImgGetter(object):
    """
    After being initialized with a directory of target images, this class will
    provide a set of methods to facilitate finding images that are similar to
    the target images.
    """

    def __init__(self, target_img_dir: str = None, target_imgs: list = None, 
                 raw_img_dir: str=None, raw_imgs: list = None, max_num_raw_imgs=None) -> None:
        """
        Initialize the SimilarImgGetter.
        """
        self.max_num_raw_imgs = max_num_raw_imgs if max_num_raw_imgs is not None else sys.maxsize
        # The model that will convert the images into feature vectors.
        self._img2vec = Img2Vec(cuda=torch.cuda.is_available(), model="resnet18", layer="default")
        # Set the target and raw images.
        self._set_target_imgs(target_img_dir, target_imgs)
        self._set_raw_imgs(raw_img_dir, raw_imgs)
        # Compute the similarity of the raw imgs to the target imgs
        self._compute_target_similarities()

    def _set_target_imgs(self, target_img_dir: str = None, target_imgs: list=None) -> None:
        self.target_imgs = target_imgs
        self.target_img_dir = target_img_dir

        # Validate input arg configuration.
        if self.target_img_dir and self.target_imgs:
            raise ValueError("Only one of target_img_dir or target_imgs can be specified")
        if not self.target_img_dir and not self.target_imgs:
            raise ValueError("One of target_img_dir or target_imgs must be specified")
        
        # If we received a directory, extract the Images from it.
        if self.target_img_dir:
            self.target_img_paths = [os.path.join(target_img_dir, f) for f in os.listdir(target_img_dir)]
            self.target_imgs = [Image.open(path) for path in self.target_img_paths]

        # Convert the images to feature vectors.
        self.target_vectors = []
        for img in tqdm(self.target_imgs, desc="Computing Target Vectors"):
            self.target_vectors.append(self._img2vec.get_vec(img))

    def _set_raw_imgs(self, raw_img_dir: str=None, raw_imgs: list = None) -> None:
        self.raw_imgs = raw_imgs
        self.raw_img_dir = raw_img_dir

        # Validate input arg configuration.
        if self.raw_img_dir and self.raw_imgs:
            raise ValueError("Only one of raw_img_dir or raw_imgs can be specified")
        if not self.raw_img_dir and not self.raw_imgs:
            raise ValueError("One of raw_img_dir or raw_imgs must be specified")

        # TODO: For now, assuming all datasets fit in memory. This is a risky assumption...
        # If we received a directory, extract the Images from it.
        if self.raw_img_dir:
            self.raw_img_paths = [os.path.join(raw_img_dir, f) for f in os.listdir(raw_img_dir)]
            self.raw_imgs = [Image.open(path) for path in self.raw_img_paths]
            for idx, img in enumerate(self.raw_imgs):
                if img.mode != "RGB":
                    print("Invalid:", self.raw_img_paths[idx], flush=True)

        self.raw_imgs = self.raw_imgs[:self.max_num_raw_imgs]

        # Convert the images to feature vectors.
        self.raw_vectors = []
        for img in tqdm(self.raw_imgs, desc="Computing Raw Vectors"):
            self.raw_vectors.append(self._img2vec.get_vec(img))

    def _compute_target_similarities(self) -> None:
        """
        For each raw image, find the nearest target image and record the similariy.
        """
        self.raw_similarities = []
        for raw_vec in tqdm(self.raw_vectors, desc="Computing Similarity..."):
            best_similarity = 0
            for target_vec in self.target_vectors:
                similarity = cosine_similarity(raw_vec.reshape((1, -1)),
                                               target_vec.reshape((1, -1)))[0][0]
                if similarity > best_similarity:
                    best_similarity = similarity
                    
            self.raw_similarities.append(best_similarity)

    def plot_similarity_disttribution(self) -> None:
        """
        Plot the distribution of similarities.
        """
        plt.hist(self.raw_similarities, bins=100)
        plt.show()

    def plot_images_in_similarity_range(self, min_similarity, max_similarity):
        """
        Plot the images that are within the specified similarity range.
        """
        similar_img_indices = [i for i, similarity in enumerate(self.raw_similarities)
                               if min_similarity <= similarity <= max_similarity]
        for plot_idx in range(5):
            plt.subplot(5, 5, plot_idx+1)
            plt.imshow(self.target_imgs[random.randint(0, len(self.target_imgs) - 1)])
            plt.axis("off")
    
        for plot_idx, img_idx in enumerate(similar_img_indices[:20]):
            plt.subplot(5, 5, plot_idx+6)
            plt.imshow(self.raw_imgs[img_idx])
            plt.axis("off")
        plt.show()

    def get_most_similar_pair(self) -> tuple:
        """
        Given a list of raw images, return the pair of images that are most similar.
        """
        closest_similarity = 0
        closest_target = None
        closest_raw = None
        for raw_idx, raw_vec in enumerate(tqdm(self.raw_vectors, desc="Computing Similarity...")):
            for target_idx, target_vec in enumerate(self.target_vectors):
                similarity = cosine_similarity(raw_vec.reshape((1, -1)),
                                               target_vec.reshape((1, -1)))[0][0]
                if similarity > closest_similarity:
                    closest_similarity = similarity
                    closest_target = Image.open(self.target_img_paths[target_idx])
                    closest_raw = self.raw_imgs[raw_idx]

        plt.subplot(1, 2, 1)
        plt.imshow(closest_target)
        plt.subplot(1, 2, 2)
        plt.imshow(closest_raw)
        plt.show()
        return closest_target, closest_raw

    def get_images_in_similarity_range(self, min_similarity: float, max_similarity: float) -> list:
        """
        Return a list of images that are above the specified similarity.
        """
        above_similarity = [i for i, similarity in enumerate(self.raw_similarities)
                            if max_similarity >= similarity >= min_similarity]
        # get the filename from the path
        return [os.path.basename(self.raw_img_paths[i]) for i in above_similarity]

    def get_image_similarity(self, img: Image) -> float:
        """
        Return the similarity of the given image to the target images.
        """
        img_vec = self._img2vec.get_vec(img)
        self.raw_similarities = []
        best_similarity = 0
        for target_vec in self.target_vectors:
            similarity = cosine_similarity(img_vec.reshape((1, -1)),
                                           target_vec.reshape((1, -1)))[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                
        return best_similarity
