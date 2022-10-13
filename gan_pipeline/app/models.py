from ast import Str
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
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
# Redis Structure
####################
"""
The redis database supports locating images by url or hash.
The following keys are used:
    Key: <gan_project>:<image_url>
    Value: {'path': <path_to_image>, 'hash': <hash_of_image>}

    Key: <gan_project>:<image_hash>
    Value: {'path': <path_to_image>, 'hash': <hash_of_image>}
"""



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
# Model Backends
####################
class GanPipelineModelBackendRedis:
    """
    Backend for GanPipelineModel using Redis.
    """
    def __init__(self, name, base_dir_path, model) -> None:
        """
        Initialize the GanPipelineModelBackendRedis.

        Note: Only need to accept the model as an argument to support backwards compatibility with
            searching through directories for files with specific names. Newer datasets use the 
            cache exclusively.
        """
        self.name = name
        self.path = base_dir_path
        self.model = model

    def has_saved_img(self, img_url: str) -> bool:
        """
        Check if an image has been saved.
        """
        # Do we have the image url in redis?
        img_val = config.REDIS_CONN.get(f'{self.name}:{img_url}')
        if img_val is not None:
            return True

        # If not in redis, fetch the image and check the hash
        img = self.fetch_image_at_url(img_url)
        hsh = sha256(img.tobytes()).hexdigest()
        hsh_val = config.REDIS_CONN.get(f'{self.name}:{hsh}')
        if hsh_val is not None:
            return True

        # Lastly, check to see if the file already exists
        img_paths = [
            os.path.join(self.model.target_path, f"{hsh}.jpg"),
            os.path.join(self.model.calibration_path, f"{hsh}.jpg"),
            os.path.join(self.model.training_path, f"{hsh}.jpg"),
        ]

        for img_path in img_paths:
            if os.path.exists(img_path):
                return True

        return False

    @lru_cache(maxsize=100)
    def fetch_image_at_url(self, img_url: str, num_retries: int=0) -> Image.Image:
        """
        Fetch an image at a url. Caches in memory and on disk before hitting network.
        """
        # Always fetch from web for now...not bandwidth efficient but that's ok
        # img_val = config.REDIS_CONN.get(f'{self.name}:{img_url}')
        # if img_val is not None:
        #     path = json.loads(img_val)['path']
        #     try:
        #         img = Image.open(path)
        #         print("  Reading from disk...")
        #         return img
        #     except FileNotFoundError:
        #         # Hmm, file is in the cache but not on disk. Must've been deleted.
        #         # Remove the cache entry and try again.
        #         config.REDIS_CONN.delete(f'{self.name}:{img_url}')
        #         return self.fetch_image_at_url(img_url, num_retries=num_retries + 1)
        # If not in redis, fetch from web
        if num_retries > 3:
            # Something funky with the url. Just return a blank image.
            return Image.new('RGB', (512, 512))
        try:
            img_response = requests.get(img_url)
            img = Image.open(BytesIO(img_response.content))
            return img
        except Exception:
            return self.fetch_image_at_url(img_url, num_retries=num_retries + 1)

    def fetch_image_if_not_saved(self, img_url: str) -> dict:
        """
        Fetch an image if it has not been saved.
        """
        if self.has_saved_img(img_url):
            return {'img': None, 'saved': False, 'error': 'duplicate', 'url': img_url}
        img = self.fetch_image_at_url(img_url)
        if img.mode != 'RGB':
            return {'img': None, 'saved': False, 'error': 'invalid img', 'url': img_url}
        return {'img': img, 'saved': True, 'error': None, 'url': img_url}

    def save_image_url_to_dir(self, img_url: str, dir: str):
        """
        Save an image to a path.
        """
        img = self.fetch_image_at_url(img_url)
        hsh = sha256(img.tobytes()).hexdigest()  # Get hash for filename
        img = self.crop_and_resize_image(img)
        img_path = os.path.join(dir, f"{hsh}.jpg")
        img.save(img_path)
        config.REDIS_CONN.set(f'{self.model.name}:{img_url}', json.dumps({'path': img_path, 'hash': hsh}))
        config.REDIS_CONN.set(f'{self.model.name}:{hsh}', json.dumps({'path': img_path, 'hash': hsh}))
    
    def save_tags_to_cache(self, tags: str):
        """
        Save tags to the cache.
        """
        # Nothing for now for redis backend
        pass

    def save_title_to_cache(self, title: str):
        """
        Save title to the cache.
        """
        # Nothing for now for redis backend
        pass

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
        
        # Create the backend
        # self.model_backend = GanPipelineModelBackendPickle(self.path, self)
        self.model_backend = GanPipelineModelBackendRedis(self.name, self.path, self)
    
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

    @property
    def candidates_path(self) -> str:
        path = os.path.join(self.path, 'Data', 'Candidates')
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
    
    def is_valid_img(self, img_url: str) -> bool:
        """
        Check if an image is valid.
        """
        try:
            img = self.model_backend.fetch_image_at_url(img_url)
            # Confirm the image is RGB
            return img.mode == 'RGB'
        except Exception:
            return False

    def save_img_url_to_calibration(self, img_url: str):
        """
        Save an image to the calibration directory.
        """
        print("Saving to calibration:", img_url)
        print("  ", self.model_backend)
        self.model_backend.save_image_url_to_dir(img_url, self.calibration_path)

    def save_img_url_to_training(self, img_url: str):
        """
        Save an image to the training directory.
        """
        self.model_backend.save_image_url_to_dir(img_url, self.training_path)

    def save_img_url_to_rejects(self, img_url: str):
        """
        Save an image to the rejects directory.
        """
        self.model_backend.save_image_url_to_dir(img_url, self.rejects_path)

    def save_img_url_to_candidates(self, img_url: str):
        """
        Save an image to the candidates directory.
        """
        self.model_backend.save_image_url_to_dir(img_url, self.candidates_path)

    def save_img_url_to_training_parallel(self, img_url, similarity_model, similarity_threshold):
        """
        Save an image to the training directory.
        """
        img = self.model_backend.fetch_image_at_url(img_url)
        similarity = similarity_model.get_image_similarity(img)
        if similarity < similarity_threshold:
            self.save_img_url_to_rejects(img_url)
            return 0
        else:
            self.save_img_url_to_training(img_url)
            # TODO: Add tags/title to cache? Useful if wanting to attempt fully automated flickr scrape.
            return 1
    
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
        max_batch_size = 100  # Number of async requests to make at once
        total_t = time.time()
        # Keep track of the current batch
        num_images = 0
        batch_size = min(max_batch_size, calibration_request.num_images - num_images)
        fetch_jobs = []
        print("Preparing initial batch of size: ", batch_size, flush=True)
        # Iterate over the flickr api until we have enough images
        for flickr_img in flickr_img_getter.get_flickr_imgs(calibration_request.search_term, time_window_days=3650, verbose=False, query_max=2000):
            fetch_jobs.append(config.QUEUE.enqueue(self.save_img_url_to_calibration, flickr_img.url))
            num_images += 1
            if len(fetch_jobs) >= batch_size:
                print("Fetching batch of size: ", batch_size, flush=True)
                batch_t = time.time()
                update_t = time.time()
                fetch_jobs_to_be_done = batch_size
                while fetch_jobs_to_be_done > 0:
                    fetch_jobs_to_be_done = len([j for j in fetch_jobs if j.get_status(refresh=True) != 'finished'])
                    time.sleep(0.1)
                    if time.time() - update_t > 2:
                        print("  ", fetch_jobs_to_be_done, "images left to fetch", flush=True)
                        update_t = time.time()

                    if time.time() - batch_t > 60:
                        print("Timeout waiting for fetch jobs to finish.", flush=True)
                        return
                
                # Print stats about the batch
                elapsed = time.time() - total_t
                rate = num_images / elapsed
                print(f"Fetched:{num_images}, Elapsed:{elapsed:.2f}, Rate:{rate:.2f}/s", flush=True)
                
                # Prep for the next batch
                batch_t = time.time()
                fetch_jobs = []
                batch_size = min(max_batch_size, calibration_request.num_images - num_images)
            
                # Do we have enough images?
                if num_images >= calibration_request.num_images:
                    return
        
        print("Shortage of images. Only found: ", num_images, flush=True)
        return

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
                if self.model_backend.has_saved_img(flickr_img.url) or not self.is_valid_img(flickr_img.url):
                    has_img += 1                        
                    continue
            
                
                # Is the image similar enough to our target set?
                img = self.model_backend.fetch_image_at_url(flickr_img.url)
                similarity = sim.get_image_similarity(img)
                if similarity < training_request.min_threshold:
                    self.save_img_url_to_rejects(flickr_img.url)
                    not_similar += 1
                    continue

                # Save the image
                self.save_img_url_to_training(flickr_img.url)
                self.model_backend.save_tags_to_cache(flickr_img.tags)
                self.model_backend.save_title_to_cache(flickr_img.title)
                num_images += 1
                pbar.update(1)
            
                if num_images >= training_request.num_images:
                    break

    def add_training_images_parallel(self, training_request: TrainingImagesRequest):
        """
        Add training images to the model.

        # Parallel: Fetched:581, Similar:200, Dupes:1, Invalid:18, Elapsed:238.57, Rate Sim:0.84/s, Rate Ovr:2.44/s
           319 rejects, 199 training
        # Serial: 329 rejects, 200 training, 115s
        
        So...for now, serial is faster. Mostly cpu-bottlenecked and parallelizing duplicates some of the
          hashing work.

        """
        sim = SimilarImgGetter(
            target_img_dir=self.target_path, 
            raw_img_dir=self.training_path,
            max_num_raw_imgs=0,
        )
        print("Constructing Flickr image getter...", flush=True)
        flickr_img_getter = FlickrImgGetter()
        max_batch_size = 100  # Number of async requests to make at once
        total_t = time.time()
        # Keep track of the current batch
        num_images_similar = 0
        num_images_total = 0
        num_duplicates = 0
        num_invalids = 0
        batch_size = min(max_batch_size, training_request.num_images - num_images_total)
        fetch_jobs = []
        print("Preparing initial batch of size: ", batch_size, flush=True)
        # Iterate over the flickr api until we have enough images
        for flickr_img in flickr_img_getter.get_flickr_imgs(training_request.search_term):
            fetch_jobs.append(config.QUEUE.enqueue(self.model_backend.fetch_image_if_not_saved, flickr_img.url))
            if len(fetch_jobs) >= batch_size:
                # print("Fetching batch of images: ", batch_size, flush=True)
                batch_t = time.time()
                update_t = time.time()
                fetch_jobs_to_be_done = batch_size
                while fetch_jobs_to_be_done > 0:
                    fetch_jobs_to_be_done = len([j for j in fetch_jobs if j.get_status(refresh=True) != 'finished'])
                    time.sleep(0.1)
                    if time.time() - update_t > 2:
                        # print("  ", fetch_jobs_to_be_done, "images left to fetch", flush=True)
                        update_t = time.time()

                    if time.time() - batch_t > 120:
                        print("Timeout waiting for fetch jobs to finish.", flush=True)
                        return
                
                # Extract the images that were actually fetched
                img_results = [j.result for j in fetch_jobs]
                new_img_dicts = [r for r in img_results if r['error'] is None]
                num_duplicates += len([r for r in img_results if r['error'] == 'duplicate'])
                num_invalids += len([r for r in img_results if r['error'] == 'invalid img'])
                num_images_total += len(new_img_dicts)
                # Process the batch to see which images are similar enough
                print("  Processing %d new images for similarity..." % len(new_img_dicts), flush=True)
                for img_dict in new_img_dicts:
                    img = img_dict['img']
                    similarity = sim.get_image_similarity(img)
                    if similarity >= training_request.min_threshold:
                        self.save_img_url_to_training(img_dict['url'])
                        num_images_similar += 1
                        if num_images_similar == training_request.num_images:
                            break
                    else:
                        self.save_img_url_to_rejects(img_dict['url'])

                # Print stats about the batch
                elapsed = time.time() - total_t
                rate_sim = num_images_similar / elapsed
                rate_ovr = num_images_total / elapsed
                print(f"Fetched:{num_images_total}, Similar:{num_images_similar}, Dupes:{num_duplicates}, Invalid:{num_invalids}, Elapsed:{elapsed:.2f}, Rate Sim:{rate_sim:.2f}/s, Rate Ovr:{rate_ovr:.2f}/s", flush=True)
                
                # Prep for the next batch
                batch_t = time.time()
                fetch_jobs = []
                batch_size = batch_size
                # batch_size = min(max_batch_size, training_request.num_images - num_images_similar)
            
                # Do we have enough images?
                if num_images_similar >= training_request.num_images:
                    return
