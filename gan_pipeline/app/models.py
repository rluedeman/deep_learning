from ast import Str
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from lib2to3.pytree import Base
import os
import pickle
import sys
import time
from typing import Tuple

from PIL import Image
from pydantic import BaseModel
import requests
from tqdm import tqdm

from gan_pipeline.app import config
from gan_pipeline.app.config import DATASETS_PATH
from gan_pipeline.img_getter.flickr_imggetter import FlickrImgGetter
from gan_pipeline.similarity import SimilarImgGetter

class GanPipelineMissingException(Exception):
    pass


####################
# Base Models
####################
class GanPipelineModelBase(BaseModel):
    """
    Pydantic model for GanPipelineModel.
    """
    name: str
    num_target_images: int = 0
    num_calibration_images: int = 0
    num_training_images: int = 0
    num_target_images: int = 0

    class Config:
        orm_mode = True


class CalibrationImageRequest(BaseModel):
    """
    Pydantic model for CalibrationImageRequest.
    """
    search_term: str
    num_images: int


class FilterCalibrationImagesRequest(BaseModel):
    """
    Pydantic model for FilterCalibrationImagesRequest.
    """
    min_threshold = 0.85
    max_threshold = 1.0
    num_images: int


class FilterCalibrationImagesResponse(BaseModel):
    """
    Pydantic model for FilterCalibrationImagesResponse.
    """
    images: list


class TrainingImagesRequest(BaseModel):
    """
    Pydantic model for TrainingImagesRequest.
    """
    search_term: str
    num_images: int
    min_threshold: float = 0.85


####################
# "ORM" Models
####################
class GanPipelineModel():
    """
    The main GanPipelineModel class.

    Instantiating this class will create a directory for the model in the DATASETS_PATH.
    """
    def __init__(self, name: str, create=False):
        self.name = name

        # Find the GanPipelineModel directory
        self.path = os.path.join(DATASETS_PATH, self.name)
        if create:
            os.makedirs(self.path, exist_ok=True)
        
        if not os.path.exists(self.path):
            raise GanPipelineMissingException(f"GanPipelineModel {self.name} does not exist.")
        
        # Load the image cache
        print("Loading image cache...")
        self.load_image_cache()
        print("  Done.")
    
    # Path helpers
    @property
    def target_path(self) -> str:
        path = os.path.join(self.path, 'Data', 'Target')
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def calibration_path(self) -> str:
        path = os.path.join(self.path, 'Data', 'Calibration')
        os.makedirs(path, exist_ok=True)
        return path
    
    @property
    def training_path(self) -> str:
        path = os.path.join(self.path, 'Data', 'Training')
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def rejects_path(self) -> str:
        path = os.path.join(self.path, 'Data', 'Rejects')
        os.makedirs(path, exist_ok=True)
        return path

    # Number of images helpers
    @property
    def num_target_images(self) -> int:
        return len(os.listdir(self.target_path))
    
    @property
    def num_calibration_images(self) -> int:
        return len(os.listdir(self.calibration_path))
    
    @property
    def num_training_images(self) -> int:
        return len(os.listdir(self.training_path))

    # Track state of img fetching
    def load_image_cache(self):
        """
        Get the image cache for the model. Keeps track of which urls have been visited and
        the hashes of the images that have been saved.
        """
        path = os.path.join(self.path, 'image_cache.pkl')
        if os.path.exists(path):
            self.image_cache = pickle.load(open(path, 'rb'))
        else:
            self.image_cache = {'urls': set(), 'hashes': set(), 'img_count_last_saved': 0}
            self.save_image_cache()
        
        if 'tags' not in self.image_cache:
            self.image_cache['tags'] = []
        if 'titles' not in self.image_cache:
            self.image_cache['titles'] = []

    def save_image_cache(self):
        """
        Snapshot the image cache for the model.
        """
        path = os.path.join(self.path, 'image_cache.pkl')
        pickle.dump(self.image_cache, open(path, 'wb'))

    def has_saved_img(self, img_url: str) -> bool:
        """
        Check if an image has been saved.
        """
        # First check to see if we've hit the url
        if img_url in self.image_cache['urls']:
            return True
        
        # Check the hashes to see if we fetched the image but just at a different url (CDN, etc.)
        img = self.fetch_image_at_url(img_url)
        hsh = sha256(img.tobytes()).hexdigest()
        if hsh in self.image_cache['hashes']:
            return True

        # Lastly, check to see if the file already exists
        img_paths = [
            os.path.join(self.target_path, f"{hsh}.jpg"),
            os.path.join(self.calibration_path, f"{hsh}.jpg"),
            os.path.join(self.training_path, f"{hsh}.jpg"),
        ]

        for img_path in img_paths:
            if os.path.exists(img_path):
                return True

        return False

    def is_valid_img(self, img_url: str) -> bool:
        """
        Check if an image is valid.
        """
        try:
            img = self.fetch_image_at_url(img_url)
            # Confirm the image is RGB
            return img.mode == 'RGB'
        except Exception:
            return False

    @lru_cache(maxsize=100)
    def fetch_image_at_url(self, img_url: str, num_retries: int=0) -> Image:
        """
        Fetch an image at a url.
        """
        if num_retries > 3:
            # Something funky with the url. Just return a blank image.
            return Image.new('RGB', (512, 512))
        try:
            img_response = requests.get(img_url)
            return Image.open(BytesIO(img_response.content))
        except Exception:
            return self.fetch_image_at_url(img_url, num_retries=num_retries + 1)

    def crop_and_resize_image(self, img: Image) -> Image:
        """
        Crop and resize an image.
        """
        # Crop the image
        new_width = min(img.width, img.height)
        new_height = new_width
        left = (img.width - new_width)/2
        top = (img.height - new_height)/2
        right = (img.width + new_width)/2
        bottom = (img.height + new_height)/2
        img = img.crop((left, top, right, bottom))
        # Resize the image
        img = img.resize((512, 512))
        return img

    def save_img_url_to_calibration(self, img_url: str):
        """
        Save an image to the calibration directory.
        """
        self._save_image_url_to_dir(img_url, self.calibration_path)

    def save_img_url_to_training(self, img_url: str):
        """
        Save an image to the training directory.
        """
        self._save_image_url_to_dir(img_url, self.training_path)

    def save_img_url_to_rejects(self, img_url: str):
        """
        Save an image to the rejects directory.
        """
        self._save_image_url_to_dir(img_url, self.rejects_path)

    def _save_image_url_to_dir(self, img_url: str, dir: str):
        """
        Save an image to a path.
        """
        img = self.fetch_image_at_url(img_url)
        hsh = sha256(img.tobytes()).hexdigest()  # Hash based on the original image
        img = self.crop_and_resize_image(img)
        img_path = os.path.join(dir, f"{hsh}.jpg")
        img.save(img_path)
        self.image_cache['urls'].add(img_url)
        self.image_cache['hashes'].add(hsh)
            
        if len(self.image_cache['hashes']) - self.image_cache['img_count_last_saved'] > 100:
            self.save_image_cache()
            self.image_cache['img_count_last_saved'] = len(self.image_cache['hashes'])

    def _save_img_to_dir(self, img: Image, dir: str):
        """
        Save an image to a path.
        """
        hsh = sha256(img.tobytes()).hexdigest()
        img_path = os.path.join(dir, f"{hsh}.jpg")
        img.save(img_path)
        self.image_cache['hashes'].add(hsh)
        if len(self.image_cache['hashes']) - self.image_cache['img_count_last_saved'] > 100:
            self.save_image_cache()
            self.image_cache['img_count_last_saved'] = len(self.image_cache['hashes'])

    def _save_tags_to_cache(self, tags: str):
        """
        Save tags to the cache.
        """
        self.image_cache['tags'].append(tags)
        self.save_image_cache()

    def _save_title_to_cache(self, title: str):
        """
        Save title to the cache.
        """
        self.image_cache['titles'].append(title)
        self.save_image_cache()
    
    #####################
    # Image Access Helpers
    #####################
    def get_calibration_image_with_id(self, img_id: str) -> Image:  
        """
        Get an image from the calibration directory.
        """
        img_path = os.path.join(self.calibration_path, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.calibration_path, f"{img_id}.png")
        return Image.open(img_path)

    def get_target_image_with_id(self, img_id: str) -> Image:
        """
        Get an image from the target directory.
        """
        img_path = os.path.join(self.target_path, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.calibration_path, f"{img_id}.png")
        return Image.open(img_path)

    ######################
    # Calibration
    ######################
    def add_calibration_images(self, calibration_request: CalibrationImageRequest):
        """
        Add calibration images to the model.
        """
        flickr_img_getter = FlickrImgGetter()
        img_urls = flickr_img_getter.get_img_urls(calibration_request.search_term)
        num_images = 0
        with tqdm(total=calibration_request.num_images) as pbar:
            for img_url in img_urls:
                if self.has_saved_img(img_url) or not self.is_valid_img(img_url):
                    continue
                else:
                    self.save_img_url_to_calibration(img_url)
                    num_images += 1
                    pbar.update(1)
                    sys.stderr.flush()
                    pbar.refresh()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    print("\r", flush=True)
                
                if num_images >= calibration_request.num_images:
                    break

    def filter_calibration_images(self, filter_request: FilterCalibrationImagesRequest) -> FilterCalibrationImagesResponse:
        """
        Filter the calibration images.
        """
        # TODO: Add support for min/max thresholds as well as limiting the number of calibration images considered
        sim = SimilarImgGetter(
            target_img_dir=self.target_path, 
            raw_img_dir=self.calibration_path,
            max_num_raw_imgs=filter_request.num_images,
        )
        return FilterCalibrationImagesResponse(images=sim.get_images_in_similarity_range(filter_request.min_threshold,
                                                                                         filter_request.max_threshold))

    ######################
    # Training Images
    ######################
    def add_training_images(self, training_request: TrainingImagesRequest):
        """
        Add training images to the model.
        """
        sim = SimilarImgGetter(
            target_img_dir=self.target_path, 
            raw_img_dir=self.training_path,
            max_num_raw_imgs=0,
        )
        print("Constructing Flickr image getter...", flush=True)
        flickr_img_getter = FlickrImgGetter()
        # Track stats
        searched_images = 0
        num_images = 0
        has_img = 0
        not_similar = 0
        print("Starting search...", flush=True)
        with tqdm(total=training_request.num_images) as pbar:
            for flickr_img in flickr_img_getter.get_flickr_imgs(training_request.search_term):
                searched_images += 1
                if searched_images % 10 == 0:
                    # Add update to the progress bar
                    pbar.set_description(f"Have Image: {has_img} | Not Similar: {not_similar} | Saved: {num_images} |||")
                    pbar.refresh()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    print("\r", flush=True)
                   

                # Do we already have the image? If so, skip it.
                if self.has_saved_img(flickr_img.url) or not self.is_valid_img(flickr_img.url):
                    has_img += 1                        
                    continue
            
                
                # Is the image similar enough to our target set?
                img = self.fetch_image_at_url(flickr_img.url)
                similarity = sim.get_image_similarity(img)
                if similarity < training_request.min_threshold:
                    self.save_img_url_to_rejects(flickr_img.url)
                    not_similar += 1
                    continue

                # Save the image
                self.save_img_url_to_training(flickr_img.url)
                self._save_tags_to_cache(flickr_img.tags)
                self._save_title_to_cache(flickr_img.title)
                num_images += 1
                pbar.update(1)
            
                if num_images >= training_request.num_images:
                    self.save_image_cache()
                    break

    def add_training_images_parallel(self, training_request: TrainingImagesRequest):
        """
        Add training images to the model but farm out image fetching and similarity comparisons to async queues.
        """
        sim = SimilarImgGetter(
            target_img_dir=self.target_path, 
            raw_img_dir=self.training_path,
            max_num_raw_imgs=0,
        )
        flickr_img_getter = FlickrImgGetter()
        with tqdm(total=training_request.num_images) as pbar:
            batch_size = 10
            batch = []
            for flickr_img in flickr_img_getter.get_flickr_imgs(training_request.search_term):
                while len(batch) < batch_size:
                    batch.append(flickr_img)

                # Process the batch
                # Sacrifice bandwidth for the sake of speed. Fetch all the images in parallel.
                print("Fetching images in parallel...", flush=True)
                image_jobs = []
                for batch_img in batch:
                    image_jobs.append(config.QUEUE.enqueue(self.async_fetch_image_and_hash, batch_img.url))
                img_jobs_to_be_done = len(image_jobs)
                while img_jobs_to_be_done > 0:
                    img_jobs_to_be_done = len([j for j in image_jobs if j.result is None])
                    time.sleep(0.1)
                    print("  Waiting for %d fetch jobs to finish..." % img_jobs_to_be_done, flush=True)
                
                # Then check for dupes.
                print("Checking for dupes...", flush=True)
                images_hashes = [j.result for j in image_jobs]
                non_dupes = [img for img, hsh in images_hashes if not self.has_img_with_hash(hsh)]

                # Finally check for similarity
                print("Checking for similarity in parallel...", flush=True)
                similarity_jobs = []
                for new_img in non_dupes:
                    similarity_jobs.append(config.QUEUE.enqueue(self.async_is_similar, new_img, sim, training_request.min_threshold))
                similarity_jobs_to_be_done = len(similarity_jobs)
                while similarity_jobs_to_be_done > 0:
                    similarity_jobs_to_be_done = len([j for j in similarity_jobs if j.result is None])
                    time.sleep(0.1)
                    print("Waiting for %d similarity jobs to finish..." % similarity_jobs_to_be_done, flush=True)
                    
                for img, is_similar in zip(non_dupes, [j.result for j in similarity_jobs]):
                    if is_similar:
                        self.save_img_url_to_training(img.url)
                        # self._save_tags_to_cache(img.tags)
                        # self._save_title_to_cache(img.title)
                        # num_images += 1
                        # pbar.update(1)
                    # else:
                    #     self.save_img_url_to_rejects(img.url)


    ##############################################################
    # Helpers for parallel training image fetching
    ##############################################################
    def async_fetch_image_and_hash(self, url: str) -> Tuple[Image.Image, str]:
        """
        Fetch an image and return its hash.
        """
        img = self.fetch_image_at_url(url)
        hsh = sha256(img.tobytes()).hexdigest()
        return img, hsh

    def has_img_with_hash(self, hsh: str) -> bool:
        """
        Check if we have an image with the given hash.
        """
        return hsh in self.image_cache['hashes']

    def async_is_similar(self, img: Image, sim_getter: SimilarImgGetter, threshold: float):
        """
        Helper function to save an image if it is similar enough to the target set.
        """
        similarity = sim_getter.get_image_similarity(img)
        if similarity < threshold:
            return False
        else:
            return True