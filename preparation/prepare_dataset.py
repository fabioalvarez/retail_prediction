from pathlib import Path
import pandas as pd
import shutil
import boto3
# import cv2
import os
from dotenv import load_dotenv
load_dotenv()

# Images
# from PIL import Image

# Initiate AWS env variables
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
BUCKET_PREFIX = os.getenv('BUCKET_PREFIX')
LOCATION_DIR= os.getenv('LOCATION_DIR') 

# Initiate paths env variables


# Get folders and files names from bucket
def get_file_folders(
    s3_client,
    bucket,
    prefix
):
    """
    This function return the list of files and folder in
    especified bucket/prefix
    corrupted images

    Parameters
    ----------
    s3_client : s3 connection
    bucket_name : str
        Name of the bucket to download images
    prefix : str
        Name of file prefix to download
        
    Returns
    -------
    folders: list
        List of folders in bucket with especified prefix

    files: list
        List of files in bucket with especified prefix
    """

    # Initiate variables
    folders, files = list(), list()
    # Define 
    default_kwargs = {
        "Bucket": bucket,
        "Prefix": prefix
    }
    # Get metadata of objects in S3 bucket
    response = s3_client.list_objects_v2(**default_kwargs)
    contents = response.get('Contents')

    # Loop over the list of objects in S3
    for result in contents:
        key = result.get('Key')
        if key[-1] == '/':
            folders.append(key)
        else:
            files.append(key)

    return folders, files

def run():
    s3_client = boto3.client("s3",
                      aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY
    )

    folders, files = get_file_folders(s3_client=s3_client, bucket=BUCKET_NAME, prefix=BUCKET_PREFIX)
    print(files[0])


def main_prepare_datasets():
    run()
    

if __name__ == '__main__':
    main_prepare_datasets()