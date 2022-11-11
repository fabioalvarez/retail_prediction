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
        os.makedirs(key_labels_dir)


def box_normalization(csv_path,
                     filename
    ):

    df_annotations = pd.read_csv(f'{csv_path}', names=["image_name", "x1", "y1", "x2", "y2","class", "image_width", "image_height"])

    normalized_coordinates = []
    starter = 0

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
        
        starter += 1

        # Create text
        normalized_coordinates.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(starter, b_center_x, b_center_y, b_width, b_height))
    
    return normalized_coordinates


def run():
    create_labels_path()
    box_normalization()



def main_prepare_labels():
    run()



if __name__ == '__main__':
    main_prepare_labels()  