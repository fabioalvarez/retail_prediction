import multiprocessing
from multiprocessing import get_context
from functools import partial
from dotenv import load_dotenv
import pandas as pd
import os
load_dotenv()

# Local paths
BASE_DIR= os.getenv('BASE_DIR')
PATH_TO_SAVE = os.getenv('PATH_TO_SAVE')
path_to_save = os.path.join(BASE_DIR, PATH_TO_SAVE)

# csv files directions
csv_paths  = os.path.join(BASE_DIR, os.getenv('csv_paths'))
txt_file   = os.path.join(csv_paths, os.getenv('txt_file'))

test_file  = os.path.join(csv_paths, os.getenv('test_file')) 
val_file   = os.path.join(csv_paths, os.getenv('val_file'))      
train_file = os.path.join(csv_paths, os.getenv('train_file')) 

# Important paths
final_images_dir = os.path.join(BASE_DIR, os.getenv('final_images_dir'))
images_dir = os.path.join(BASE_DIR, os.getenv('images_dir'))
split_dir = os.path.join(BASE_DIR, os.getenv('split_dir'))


def create_labels_path():

    # List of dirs to create
    dir_keys = ["train", "val", "test"]

    for key in dir_keys:
        key_labels_dir = os.path.join(split_dir,'labels',key)

        if not os.path.exists(key_labels_dir):
            os.makedirs(key_labels_dir)



def box_normalization(csv_paths, subset, filename):

    df_annotations = pd.read_csv(f'{csv_paths}annotations_{subset}.csv', names=["image_name", "x1", "y1", "x2", "y2","class", "image_width", "image_height"])

    normalized_coordinates = []
    obj_class = 0

    for i in df_annotations.loc[df_annotations['image_name'] == filename].values:
        
        b_center_x = (i[1] + i[3]) / 2 
        b_center_y = (i[2] + i[4]) / 2
        b_width    = (i[3] - i[1])
        b_height   = (i[4] - i[2])

        # Normalise the co-ordinates by the dimensions of the image
        image_w = i[6]
        image_h= i[7]
        image_c = i[5]
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h

        # Create text
        normalized_coordinates.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(obj_class, b_center_x, b_center_y, b_width, b_height))
    
    return normalized_coordinates

def normalized_text(csv_paths, split_dir, subset, filename):
    sub_path="labels"

    # Get list of box coordinates by image
    normalized_coordinates = box_normalization(csv_paths= csv_paths, subset=subset, filename= filename)

    # Path to save images
    key_path = filename.split("_")[0]
    name = filename.split(".")[0] + ".txt"

    # Full path
    txt_path = os.path.join(split_dir,sub_path,key_path,name)

    # Append list and create txt
    with open(txt_path, 'w') as f:
        f.write("\n".join(normalized_coordinates))

def get_images_list(csv_paths, subset):
    # List of image names
    df = pd.read_csv(f'{csv_paths}annotations_{subset}.csv', names=["image_name", "x1", "y1", "x2", "y2","class", "image_width", "image_height"])

    # Extract unique names
    images_list = list(df["image_name"].unique())

    return images_list


def run():

    create_labels_path()

    # Process for test
    subset = "train"
    images_list = get_images_list(csv_paths=csv_paths, subset=subset)
    # Instanciate multiprocessing pool
    pool = multiprocessing.get_context("fork").Pool()
    func = partial(normalized_text,csv_paths ,split_dir, subset)
    pool.map(func, images_list)
    pool.close()
    pool.join()

    subset = "val"
    images_list = get_images_list(csv_paths=csv_paths, subset=subset)
    # Instanciate multiprocessing pool
    pool = multiprocessing.get_context("fork").Pool()
    func = partial(normalized_text,csv_paths ,split_dir, subset)
    pool.map(func, images_list)
    pool.close()
    pool.join()

    subset = "test"
    images_list = get_images_list(csv_paths=csv_paths, subset=subset)
    # Instanciate multiprocessing pool
    pool = multiprocessing.get_context("fork").Pool()
    func = partial(normalized_text,csv_paths ,split_dir, subset)
    pool.map(func, images_list)
    pool.close()
    pool.join()


def main_prepare_labels():
    run()
    

if __name__ == '__main__':
    main_prepare_labels()  