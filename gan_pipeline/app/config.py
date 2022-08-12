from datetime import datetime
import os
import pathlib

from redis import Redis
from rq import Queue

DATASETS_PATH = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), 'datasets', 'gan_pipeline')
REDIS_CONN = Redis(host="redis-service")
QUEUE = Queue(connection=REDIS_CONN)


def get_gan_pipeline_datasets():
    return os.listdir(DATASETS_PATH)