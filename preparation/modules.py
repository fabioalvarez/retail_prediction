from pathlib import Path
import pandas as pd
import shutil
import cv2
import os

# Images
from PIL import Image






        


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