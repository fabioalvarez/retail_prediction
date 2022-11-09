from pathlib import Path
import pandas as pd
import shutil
import cv2
import os

# Images
from PIL import Image





# Download files
def download_files(
    s3_client,
    bucket_name:str,
    local_path:str,
    file_names:list,
    folders:list,
    prefix:str
):
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

# Check corrupted files
def check_bad_files(images_dir:str):
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

# Plot bounding boxes
def plot_bounding_boxes(
    images_dir:str,
    df:pd.DataFrame,
    column:str,
    final_images_dir:str
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

# Move images
def mov_images(
    images_dir: str,
    split_dir:str,
    images_list:list
):

    for image_name in images_list:
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


        


############################







def labels_yolo_format(df:pd.DataFrame, split_dir:str, ):

    for index, row in df.iterrows():
        
        # Extract class and image name
        class_id = "0" if row['class'] == "object" else "N/A"
        image_name =  row['image_name'].split(".")[0]

        # Transform coordinates to Yolo format
        width  = (row['x2'] - row['x1'])
        heigth = (row['y2'] - row['y1'])
        x_center = (row['x2'] + row['x1'])/2
        y_center = (row['x2'] + row['y1'])/2

        # Normalize coordinates
        x_center = x_center/row['image_width']
        y_center = y_center/row['image_height']
        width    = width/row['image_width']
        heigth   = heigth/row['image_height']

        # Create text
        text = ("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, x_center, y_center, width, heigth))

        # Create full path to save labels
        key_path = row['image_name'].split('_')[0]
        key_labels_dir = os.path.join(split_dir,'labels',key_path)

        # Create paths for txt files
        file_path = os.path.join(key_labels_dir, image_name+".txt")

        # Creat txt files and paths
        if not os.path.exists(key_labels_dir):
            os.makedirs(key_labels_dir)

        # Write file
        with open(file_path, "a") as f:
            f.write(text+'\n')
            f.close()