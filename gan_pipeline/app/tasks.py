import requests
import time

from gan_pipeline.app import config

def count_words_at_url(url):
    time.sleep(1)
    print("Enqueued:", len(config.QUEUE))
    resp = requests.get(url)
    return len(resp.text.split())




