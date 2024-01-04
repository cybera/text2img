FROM cschranz/gpu-jupyter:v1.4_cuda-11.0_ubuntu-20.04_python-only

COPY /requirements.txt /

RUN pip install --upgrade pip &&\
    pip install -r /requirements.txt
