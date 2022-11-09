FROM python:3.9 as base

WORKDIR  /src

RUN pip3 install --upgrade pip

COPY ./requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /src/requirements.txt

COPY . /src/preparation/