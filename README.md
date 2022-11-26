# retail_prediction
Predict empty spaces in supermarkets

# Preparation

## Download files from 

```
wget http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
```

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
model’s head. YOLOv5s6 backbone consists of 10 layers, who will be fixed by the ‘freeze’ 
argument.

train scrips:
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
- project— name of the project
- name — name of the run

If ‘project’ and ‘name’ arguments are supplied, the results are automatically saved there.
Else, they are saved to ‘runs/train’ directory. 

## Docker
```
docker build -t yolov5_fa .
```
### Docker training

```
docker run --rm --net host --gpus all -it \
    -v /home/fabioalvarez/retail_prediction/training:/home/src/app \
    -v /home/eudesz/final_project/data:/home/src/dataset \
    -v /home/fabioalvarez/retail_prediction/data:/home/src/data \
    --workdir /home/src \
    preparation_fa \
    bash
```

### Docker model

```
docker build -t model_fa .
```

```
docker run --rm --net host --gpus all -it \
    -v /home/fabioalvarez/retail_prediction/yolov5:/home/src/app \
    -v /home/fabioalvarez/retail_prediction/data:/home/src/data \
    -v /home/fabioalvarez/retail_prediction/model:/home/src/model \
    --workdir /home/src \
    model_fa \
    bash
```


### Tmux
```
tmux new -t fabio_train
```

### Start trianing

Run this inside yolov5 folder

```
python3 train.py --img 416 --batch 4 --epochs 3 \
    --data /home/src/app/yolo_config/retail_config.yaml \
    --weights yolov5s.pt --cache --project /home/src/data \
    --name retail
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