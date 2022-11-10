from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
import pandas as pd
import shutil
import boto3
# import cv2
import os
load_dotenv()

# Initiate AWS env variables
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
BUCKET_PREFIX = os.getenv('BUCKET_PREFIX')

# Local paths
BASE_DIR= os.getenv('BASE_DIR')
PATH_TO_SAVE = os.getenv('PATH_TO_SAVE')
path_to_save = os.path.join(BASE_DIR, PATH_TO_SAVE)

# csv files directions
csv_paths  = os.getenv('csv_paths') 
txt_file   = os.path.join(csv_paths, os.getenv('txt_file'))
test_file  = os.path.join(csv_paths, os.getenv('test_file')) 
val_file   = os.path.join(csv_paths, os.getenv('val_file'))      
train_file = os.path.join(csv_paths, os.getenv('train_file')) 

# Important paths
final_images_dir = os.path.join(BASE_DIR, os.getenv('final_images_dir'))
images_dir = os.path.join(BASE_DIR, os.getenv('images_dir'))
split_dir = os.path.join(BASE_DIR, os.getenv('split_dir'))

# Initiate paths env variables


# Get folders and files names from bucket
def get_file_folders(
    s3_client,
    bucket,
    prefix
):
    """
    This function return the list of files and folder in especified
    bucket/prefix
    
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
    folders, files = [], []
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
    
# Download files
def download_files(
    s3_client,
    bucket_name:str,
    file_names:list,
    folders:list,
    prefix:str,
    path_to_save,
    
):
    """
    This function download files from AWS with a especified
    bucket and prefix

    Parameters
    ----------
    s3_client : s3 connection bucket_name : str
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
    path_to_save = Path(path_to_save)

    for folder in folders:
        folder_path = Path.joinpath(path_to_save, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = file_name.replace(prefix+'/', '')
        file_path = Path.joinpath(path_to_save, file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )

# Check corrupted files
def check_bad_files(images_dir:str):
    """
    This function return the list of corrupted images

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

# Move images # check bad files before send bad images
def mov_images(
    images_dir: str,
    split_dir:str,
):

    # First check for bad files
    bad_files = check_bad_files(images_dir)

    # List images to move
    images_list = os.listdir(images_dir)

    for image_name in images_list:
        if image_name not in bad_files:

            # Get the key path of the image (train/test/val)
            key_path = image_name.split('_')[0]

            # Creat full path to save the image
            src_path = os.path.join(images_dir, image_name)
            key_img_dir = os.path.join(split_dir,'images',key_path)
            dst_img_dir = os.path.join(key_img_dir, image_name)

            # Create paths for images and move images
            if not os.path.exists(key_img_dir):
                os.makedirs(key_img_dir)
            
            if not os.path.exists(dst_img_dir):
                shutil.move(src_path, dst_img_dir)

# Concatenate dataframes
def concatenate_csv(txt_file,
              test_file,
              train_file,
              val_file
    ):

    # Read txt file
    with open(txt_file) as f:
        txt = f.read()

    # Extract column names
    column_names = txt.split(":")[1].strip()
    columns = column_names.split(',')

    # Read csv and check null values and add column to identify dataset
    test_df = pd.read_csv(test_file, index_col=False, names=columns)
    test_df['set'] = 'test'

    # Read csv and check null values and add column to identify dataset
    train_df = pd.read_csv(train_file, index_col=False, names=columns)
    train_df['set'] = 'train'

    # Read csv and check null values and add column to identify dataset
    val_df = pd.read_csv(val_file, index_col=False, names=columns)
    val_df['set'] = 'val'

    # Unificate datasets
    df = pd.concat([test_df, train_df, val_df], axis=0, ignore_index=True)

    return df

# Plot bounding boxes
def plot_bounding_boxes(
    final_images_dir:str,
    df:pd.DataFrame,
    images_dir:str,
    column:str,
):

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

def run():
    try:
        s3_client = boto3.client("s3",
                                 aws_access_key_id=ACCESS_KEY,
                                aws_secret_access_key=SECRET_KEY)
    except:
        raise ValueError("Cannot conect with AWS")

    print("Connected succesfully")
    folders, files = get_file_folders(s3_client=s3_client,
                                        bucket=BUCKET_NAME, 
                                        prefix=BUCKET_PREFIX)
    
    print("Got folder and file list in bucket")
    download = download_files(s3_client=s3_client,
                        bucket_name=BUCKET_NAME,
                        prefix=BUCKET_PREFIX,
                        folders=folders,
                        file_names=files,
                        path_to_save=path_to_save
    )
    
    print("Files downloaded")
    mov_img = mov_images(images_dir,
                        split_dir)

    print("Files splitted and organized with Yolo format")

def main_prepare_datasets():
    run()

if __name__ == '__main__':
    main_prepare_datasets()