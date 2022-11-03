from pathlib import Path
import shutil
import pandas as pd
import cv2
import os

# Images
from PIL import Image

def get_file_folders(s3_client, bucket, prefix):
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
    folders = list()
    files = list()
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

def download_files(s3_client, bucket_name:str, local_path:str, file_names:list, folders:list, prefix:str):
    """
    This function return the list of
    corrupted images

    Parameters
    ----------
    s3_client : s3 connection
    bucket_name : str
        Name of the bucket to download images
    local_path : str
        Local path to download images
    local_path : list
        List of file names to download
    folders : list
        List of folder names to download
    prefix : str
        Name of file prefix to download
        
    Returns
    -------
    Download especified images in path
    """

    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = file_name.replace(prefix+'/', '')
        file_path = Path.joinpath(local_path, file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )
        

def check_bad_files(images_dir):
    """
    This function return the list of
    corrupted images

    Parameters
    ----------
    images_dir : str
        Path of images

    Returns
    -------
    bad_files : list
        List of bad image names
    """
    bad_files = []
    for filename in os.listdir(images_dir):
        path = os.path.join(images_dir,filename)
        try:
            img = Image.open(path)
            img.verify()
        except (IOError, SyntaxError) as e:
            bad_files.append(filename)
            
    return bad_files


def plot_bounding_boxes(images_dir:str, df:pd.DataFrame, column:str, final_images_dir:str):

    count = 0
    image_list = os.listdir(images_dir)

    for image_name in image_list:
        print(image_name)
        # Create paths
        image_path = os.path.join(images_dir, image_name)
        final_dir =  os.path.join(final_images_dir, image_name)

        # Select boundind boxes for especified image
        bounding_list = df[df[column] == image_name]

        # Open image
        image = cv2.imread(image_path)

        for _, row in bounding_list.iterrows():
            start_point = (int(row["x1"]), int(row["y1"])) 
            end_point = (int(row["x2"]), int(row["y2"]))

            cv2.rectangle(image, start_point, end_point, (36,255,12), 2)

        cv2.imwrite(final_dir, image)
        count += 1
        if count == 10:
            break


def mov_images(images_dir: str, split_dir:str, images_list:list):

    # Create paths
    for image_name in images_list:
        # Get the key path of the image (train/test/val)
        print(image_name)
        key_path = image_name.split('_')[0]

        # Creat full path to save the image
        src_path = os.path.join(images_dir, image_name)
        key_dir = os.path.join(split_dir, key_path)
        dst_dir = os.path.join(key_dir, image_name)

        if not os.path.exists(key_dir):
            os.mkdir(key_dir)
        
        if not os.path.exists(dst_dir):
            shutil.move(src_path, dst_dir)