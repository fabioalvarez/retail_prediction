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