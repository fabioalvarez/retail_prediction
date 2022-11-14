# Preparation

## Image and labels directories structure
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

## Data configuration files
The configurations for the training are divided to three YAML files, which are provided
with the repo itself. We will customize these files depending on the task, to fit our 
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
    - These architectures are suitable for training with image size of 640*640 pixels
    - YOLOv5n (nano)
    - YOLOv5s (small)
    - YOLOv5m (medium)
    - YOLOv5l (large)
    - YOLOv5x (extra large)


```
yolov5
├── data
│   ├── retail_data.yaml
│   ├── test
|   ├── hyps
|   |   ├── hyp.scratch-low.yaml
|   |   ├── hyp.scratch-med.yaml
|   |   ├── hyp.scratch-high.yaml
├── models
│   ├── yolov5l.yaml
│   ├── yolov5m.yaml
│   ├── yolov5n.yaml
│   ├── yolov5s.yaml
│   └── lyolov5x.yaml
```


## Transfer Learning
Models are composed of two main parts: the backbone layers which serves as a feature extractor, 
and the head layers which computes the output predictions. To further compensate for a small
dataset size, we’ll use the same backbone as the pretrained COCO model, and only train the
model’s head. YOLOv5s6 backbone consists of 12 layers, who will be fixed by the ‘freeze’ 
argument.





Download files from 

```
wget http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
```


```
docker build -t preparation_fa .
docker build -t preparation -f preparation/Dockerfile .
```

```
docker run --rm --net host -it\
    -v $(pwd):/home/src/app \
    -v /home/fabioalvarez/retail_prediction/data:/home/src/data \
    -v /home/fabioalvarez/retail_prediction/.env:/home/src \
    --workdir /home/src \
    preparation \
    bash
```

```bash
$ docker run --rm --net host --gpus all -it \
    -v /home/fabioalvarez/retail_prediction/training:/home/src/app \
    -v /home/fabioalvarez/retail_prediction/data:/home/src/data \
    --workdir /home/src \
    preparation_fa \
    bash
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