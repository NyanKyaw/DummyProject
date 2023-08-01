# %% package import
import pandas as pd
import os
import shutil
import math

# Defining directories
# current_dir = os.getcwd()
# csv_file_dir = current_dir + "/image_data/data/export1/_annotations.csv"
csv_file_dir = "Tutorial_3/data/_annotations.csv"
image_dir = "Tutorial_3/data/export/"

# Making csv dataframe containing all annotations
annot_df = pd.read_csv(csv_file_dir)

# Create folders
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_folder('Tutorial_3/data/train/images')
create_folder('Tutorial_3/data/train/labels')
create_folder('Tutorial_3/data/val/images')
create_folder('Tutorial_3/data/val/labels')
create_folder('Tutorial_3/data/test/images')
create_folder('Tutorial_3/data/test/labels')


# Function for calculating bounding boxes
def make_bbox(width, height, xmin, ymin, xmax, ymax):
    x_center = ((xmax + xmin) / 2) / width
    y_center = ((ymax + ymin) / 2) / height
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height
    return [x_center, y_center, width, height]


labels_dict = {'pedestrian': 0,
               'car': 1,
               'truck': 2,
               'trafficLight-Red': 3,
               'trafficLight-Green': 4,
               'trafficLight-Yellow': 5,
               'trafficLight': 6,
               'biker': 7,
               'trafficLight-RedLeft': 8,
               'trafficLight-GreenLeft': 9,
               'trafficLight-YellowLeft': 10
               }


# Grouping images by image ID
image_groups = annot_df.groupby('filename')

# Defining partitions 80 Training 10 Validation 10 testing
# num_groups = len(image_groups)
# t_index = math.ceil(num_groups * 0.8)
# v_index = t_index + math.ceil(num_groups * 0.1)

num_groups = 2500
t_index = math.ceil(num_groups * 0.8)
v_index = t_index + math.ceil(num_groups * 0.1)

# Counter for partitioning dataset
count = 0
# For unique image in dataset
for image_ID, group in image_groups:
    count += 1
    if count < t_index:
        image_dest_dir = 'Tutorial_3/data/train/images'
        label_dir = 'Tutorial_3/data/train/labels'
    elif count < v_index:
        image_dest_dir = 'Tutorial_3/data/val/images'
        label_dir = 'Tutorial_3/data/val/labels'
    else:
        image_dest_dir = 'Tutorial_3/data/test/images'
        label_dir = 'Tutorial_3/data/test/labels'

    image_source_dir = image_dir + image_ID

    # Checking if image exists
    if os.path.isfile(image_source_dir):
        # Copy accross image from source directory to image directory
        shutil.copy(image_source_dir, image_dest_dir)

        # For each data point relating to image
        for index, row in group.iterrows():
            width, height, label, xmin, ymin, xmax, ymax = row['width'], row['height'], row['class'], row['xmin'], row[
                'ymin'], row['xmax'], row['ymax']

            # If data point describes a car, truck or pedestrian object:
            if label in labels_dict:
                # Making text file for data point
                x_center, y_center, width, height = make_bbox(width, height, xmin, ymin, xmax, ymax)
                result = f"{labels_dict[label]} {x_center} {y_center} {width} {height}"

                # Writing to text file
                label_name = f"{image_ID}" + ".txt"
                label_name = label_name.replace('.jpg', '') # remove .jpg from filename or else YOLO can't find it
                file_path = label_dir + "/" + label_name 

                with open(file_path, 'a+', encoding="utf-8") as f:
                    f.write(result + "\n")

    if count == num_groups:
        exit()

