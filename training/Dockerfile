FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04 as base

# Install some packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    vim \
    wget \
    curl

RUN apt-get -y update
RUN apt-get -y install git

# Add a non-root user
RUN useradd -ms /bin/bash app
USER app

# Setup some paths
ENV PYTHONPATH=/home/src/.local/lib/python3.8/site-packages:/home/src
ENV PATH=$PATH:/home/src/.local/bin

# Install the python packages for this new user
ADD requirements.txt .
RUN pip3 install -r requirements.txt

# PyTorch GPU 1.10
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# Copy info inside 
ADD . /home/src

# Create working directory
WORKDIR /home/src

# Clone yolov5
RUN git clone https://github.com/ultralytics/yolov5 /home/src/yolov5