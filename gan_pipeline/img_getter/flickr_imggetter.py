from hashlib import sha256
from io import BytesIO

import flickrapi
from PIL import Image
import requests
from typing import Tuple

from gan_pipeline.img_getter import config


class ImgGetter():
    """
    Parent class for ImgGetter classes.
    """
    def process_img(self, img: Image) -> Image:
        """
        Processes the image.
        """
        return img
        width, height = img.size
        # Crop Square
        new_width = min(width,height)
        new_height = new_width
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        img = img.crop((left, top, right, bottom))
        return img


class FlickrImgGetter(ImgGetter):
    """
    Class for getting images from Flickr.
    """
    def __init__(self):
        self.flickr_api = flickrapi.FlickrAPI(config.FLICKR_API_KEY, config.FLICKR_API_SECRET, cache=True)

    def get_img_urls(self, search_term: str) -> Tuple[str, str]:
        photos = self.flickr_api.walk(
            # tag_mode='all',
            # tags=search_term,
            text=search_term,
            extras='url_c,license',
            per_page=100,
            sort='relevance',
            min_height=512,
            min_width=512,
        )
        for photo in photos:
            url = photo.get('url_c')
            if url:
                yield url

    def get_images(self, search_term: str, num_images: int) -> Tuple[Image.Image, str]:
        photos = self.flickr_api.walk(
            # tag_mode='all',
            # tags=search_term,
            text=search_term,
            extras='url_c,license',
            per_page=100,
            sort='relevance',
            min_height=512,
            min_width=512,
        )

        for photo in photos:
            url = photo.get('url_c')
            if url:
                response = requests.get(url)
                hsh = sha256(response.content).hexdigest()
                img = Image.open(BytesIO(response.content))
                img.load()
                img = self.process_img(img)
                if img:
                    yield (img, hsh)
            

