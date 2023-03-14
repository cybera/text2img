FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY /requirements.txt /

RUN pip install --upgrade pip &&\
    pip install -r /requirements.txt
