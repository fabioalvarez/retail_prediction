FROM python:3.9 as base

WORKDIR  /home

RUN pip3 install --upgrade pip

COPY ./requirements.txt /home/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /home/requirements.txt

COPY . /home/app/src