# from api.redis import settings
from app.detect import parse_opt
from app.detect import main_detect
from dotenv import load_dotenv
import numpy as np
import yaml
import json
import os
import time
# import redis

load_dotenv()

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
# db = redis.Redis(host=settings.REDIS_IP,
#                  port=settings.REDIS_PORT,
#                  db=settings.REDIS_DB_ID)


# Load env variables
database = os.getenv('DATABASE')
project = os.getenv('PROJECT')
weights = os.getenv('WEIGTHS')
save_txt = os.getenv('SAVE_TXT')
source =os.getenv('SAVE_TXT')


# Load preset variables preset from the ArgParse function
args = parse_opt()

# Change values
args.save_conf = True
args.save_txt = True
args.project = project
# args.weights = weights


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # Update image path and add path to args
    image_path = os.path.join(source, image_name)

    # Update name and source arguments
    args.name = image_name
    args.source = image_path

    # Predict from image
    run = main_detect(args)

    return class_name, pred_probability


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    # while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        # job = db.brpop(settings.REDIS_QUEUE)[1]

        # job = json.loads(job.decode("utf-8"))

        # image_name = job["image_name"]
        # job_id = job["id"]

        # #   2. Run your ML model on the given data
        # prediction, score = predict(image_name)

        # #   3. Store model prediction in a dict with the following shape:
        # rpse = { "prediction": str(prediction), "score": float(score)}

        # #   4. Store the results on Redis using the original job ID as the key
        # db.set(job_id, json.dumps(rpse))

        # # Don't forget to sleep for a bit at the end
        # time.sleep(settings.SERVER_SLEEP)



if __name__ == '__main__':
    # Now launch process
    print("Launching ML service...")
    predict("zidane.jpg")
    #classify_process()