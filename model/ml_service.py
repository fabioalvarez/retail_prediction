from scripts.detect_voids import main_detect_voids
from dotenv import load_dotenv
import numpy as np
import yaml
import json
import os
import time
import redis

# Import modules
import settings

load_dotenv()

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(host=settings.REDIS_IP,
                 port=settings.REDIS_PORT,
                 db=settings.REDIS_DB_ID)


def predict_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        job = db.brpop(settings.REDIS_QUEUE)[1]

        job = json.loads(job.decode("utf-8"))

        image_name = job["image_name"]
        job_id = job["id"]

        #   2. Run your ML model on the given data
        status = main_detect_voids(image_name)

        #   3. Store model prediction in a dict with the following shape:
        rpse = {"status": str(status)}

        #   4. Store the results on Redis using the original job ID as the key
        db.set(job_id, json.dumps(rpse))

        # Don't forget to sleep for a bit at the end
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    predict_process()