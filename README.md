# retail_prediction
Predict empty spaces in supermarkets



# preparation

```
docker build -t preparation .
```

```
docker run --rm --net host -it\
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    preparation \
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


## Move images

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
│   │   ├── test
```