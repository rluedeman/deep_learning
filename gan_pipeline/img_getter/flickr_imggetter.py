from datetime import datetime, timedelta
from hashlib import sha256
from io import BytesIO
import time

import flickrapi
from PIL import Image
import requests
from typing import Iterable, Tuple

from gan_pipeline.img_getter import config


class ImgGetter():
    """
    Parent class for ImgGetter classes.
    """
    def process_img(self, img: Image) -> Image:
        """
        Processes the image.
        """
        # Nothing to process for now.
        return img


class FlickrImg(object):
    def __init__(self, url, title, tags):
        self.url = url
        self.title = title
        self.tags = tags



class FlickrImgGetter(ImgGetter):
    """
    Class for getting images from Flickr.
    """
    def __init__(self):
        self.flickr_api = flickrapi.FlickrAPI(config.FLICKR_API_KEY, config.FLICKR_API_SECRET, cache=True)

    def get_flickr_imgs(self, search_term: str, time_window_days=2, query_max=100, verbose=True) -> Iterable[FlickrImg]:
        """
        A generator that will return an iterable of FlickrImgs matching the search_term.
        """
        min_time = datetime(2010, 1, 1)
        max_time = datetime(2022, 1, 1)
        cur_time = min_time
        time_window = timedelta(days=time_window_days)
        while cur_time < max_time:
            start_date = time.mktime(cur_time.timetuple())
            end_date = time.mktime((cur_time + time_window).timetuple())
            if verbose:
                print("****************** Date:", cur_time, cur_time + time_window)
            photos = self.flickr_api.walk(
                # tag_mode='all',
                # tags=search_term,
                text=search_term,
                extras='url_c,license,tags,machine_tags',
                per_page=100,
                sort='relevance',
                min_height=512,
                min_width=512,
                min_upload_date=start_date,
                max_upload_date=end_date,
            )
            num_photos_in_query = 0
            for photo in photos:
                num_photos_in_query += 1
                url = photo.get('url_c')
                if url:
                    yield FlickrImg(url, photo.get('title'), photo.get('tags'))

                # Limit amount per query to minimize duplicate images.
                if num_photos_in_query >= query_max:
                    break

            cur_time += time_window

    def _get_img_urls(self, search_term: str) -> Tuple[str, str]:
        photos = self.flickr_api.walk(
            # tag_mode='all',
            # tags=search_term,
            text=search_term,
            extras='url_c,license,tags,machine_tags',
            per_page=100,
            sort='relevance',
            min_height=512,
            min_width=512,
        )
        for photo in photos:
            url = photo.get('url_c')
            if url:
                yield url

    def _get_images(self, search_term: str, num_images: int) -> Tuple[Image.Image, str]:
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
            

