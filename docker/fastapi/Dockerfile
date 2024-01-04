FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

WORKDIR /app

COPY /requirements.txt /

RUN pip install --upgrade pip &&\
    pip install -r /requirements.txt
