# Welcome
Welcome to the retail object prediction project. The goal of this project is to find
empty spaces in super market shelves, so, managers will fill those with products in
the rigth moment.

This project is divided in different topics.

- Training.
- Service deployment.
- Tests.


# Training
The goal of this topic is to train the yolov5 model and get the weigths to predict empty
spaces in shelves.

To get this result we need:
- Prepare dataset and labels: 
- Set YoloV5 data configurations files.
- Train


## Preparation

To start this process, you need to build the preparation image. You need to run the next line
inside trainig folder.

```
docker build -t api_test .
```

After that you need to run the container. That's all you need to start the training process (XXXXXXX)

```
docker run --rm --net host --gpus all -it \
    -v /home/fabioalvarez/retail_prediction/training:/home/src/app \
    -v /home/eudesz/final_project/data:/home/src/dataset \
    -v /home/fabioalvarez/retail_prediction/data:/home/src/data \
    --workdir /home/src \
    preparation_fa \
    bash
```

The explanation of preparation process is as follows.

### Download dataset
Initially we are not going to download the dataset due to EC2 space limit.
So we will map the dataset from the Eudes EC2 account.

```
wget http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
```

## YoloV5 required structure

### Image directories structure
Yolo V5 expects a folder architecture to infer class and test/train/val datasets:

```
data
├── penguins
│   ├── images
│   │   ├── train
│   │   ├── validation
│   │   ├── test
│   ├── labels
│   │   ├── train
│   │   ├── validation
│   │   └── test
```

The script in charge to create this structure is 'prepare_dataset.py' located in
'/training/preparation/prepare_dataset.py'.

### Labels structure
YOLO v5 expects annotations for each image in form of a .txt file where each line
of the text file describes a bounding box.

The annotations looks like:

'''
0  0.480 0.631 0.692 0.713
0  0.741 0.522 0.314 0.933
27 0.785 0.506 0.390 0.151
'''

There are 3 objects in total (2 persons and one tie). Each line represents one of these objects.
The specification for each line is as follows.

- One row per object.
- Each row is: class | x_center | y_center | width | height |
- Box coordinates must be normalized by the dimensions of the image (i.e. have values between 0 and 1).
- Class numbers are zero-indexed (start from 0).

The script in charge to create this structure is 'prepare_labels.py' located in
'/training/preparation/prepare_labels.py'.


## Data configuration files
The configurations for the training are divided to three YAML files, which are provided
with the YoloV5 repo itself. We will customize these files depending on the task, to fit our 
desired needs.

1. data-configurations file: describes datasets parameters
    - train/val/test paths
    - nc: number of classes
    - names: classes names
    
2. hyperparameter config file: defines the hyperparameters for the training
    - learning rate
    - momemtum
    - losses
    - augmentation
    - etc

3. models-configuration file: dictates the model architecture.
   These architectures are suitable for training with image size of 640*640 pixels
    - YOLOv5n (nano)
    - YOLOv5s (small)
    - YOLOv5m (medium)
    - YOLOv5l (large)
    - YOLOv5x (extra large)

Locations:

```
yolov5
├── data
|   ├── hyps                                # Hyper parameter configuration file
|   |   ├── hyp.scratch-low.yaml
|   |   ├── hyp.scratch-med.yaml
|   |   ├── hyp.scratch-high.yaml
├── models                                  # Model architecture file
│   ├── yolov5l.yaml
│   ├── yolov5m.yaml
│   ├── yolov5n.yaml
│   ├── yolov5s.yaml
│   └── lyolov5x.yaml
```

```
training
├── analisis
├── preparation
├── yolo_config
│   └── retail_config.yaml                  # Data configuration file
```

## Transfer Learning
Models are composed of two main parts: the backbone layers which serves as a feature extractor, 
and the head layers which computes the output predictions. To further compensate for a small
dataset size, we’ll use the same backbone as the pretrained COCO model, and only train the
model’s head. YOLOv5s6 backbone consists of 10 layers, who will be fixed by the ‘freeze’ 
argument.

train script example:
```
python train.py --batch 32 --epochs 100 --data 'yolov5/data/retail_data.yaml'
-- weights 'yolov5s.pt' --cache --freeze 10  --project retail_prediction --name 'feature_extraction'
```

- batch — batch size (-1 for auto batch size). Use the largest batch size that your hardware allows for.
- epochs — number of epochs.
- data — path to the data-configurations file.
- weights — path to initial weights. COCO model will be downloaded automatically.
- cache — cache images for faster training.
- img — image size in pixels (default — 640).
- freeze — number of layers to freeze
- project— direction to save weights.
- name — weigths folder name

If ‘project’ and ‘name’ arguments are supplied, the results are automatically saved there.
Else, they are saved to ‘runs/train’ directory. 

### Start trianing

Run this inside yolov5 folder in the preparation container

```
python3 train.py --img 416 --batch 4 --epochs 3 \
    --data /home/src/app/yolo_config/retail_config.yaml \
    --weights yolov5s.pt --cache --project /home/src/data/weights \
    --name retail
```






### Docker model

```
docker build -t model_fa .
```

```
docker run --rm --net host --gpus all -it \
    -v /home/fabioalvarez/retail_prediction/yolov5:/home/app/src/service \
    -v /home/fabioalvarez/retail_prediction/data:/home/app/src/data \
    -v /home/fabioalvarez/retail_prediction/model:/home/app/src/model \
    --workdir /home/app/src \
    model_fa \
    bash
```


```
docker run --rm --net host --gpus all -it \
    --workdir /home/src \
    api_fa \
    bash
```


```bash
$ docker-compose up --build -d
```

### Tmux
```
tmux new -t fabio_train
```



# Check problems
```
DOCKER_BUILDKIT=0  docker build .
```





# api
```
.
├── app
│   ├── __init__.py
│   └── main.py
├── Dockerfile
└── requirements.txt

```


# Docker build image

```
docker build -t api_test .
```