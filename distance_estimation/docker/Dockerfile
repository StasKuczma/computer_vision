FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
    screen \
    git \
    vim \
    python3-pip \
    htop \
    ffmpeg \
    libsm6 \
    libxext6 \
    feh \
    wget


WORKDIR /root/
RUN git clone https://github.com/WongKinYiu/yolov9.git 


RUN pip install python-dateutil==2.8.2

WORKDIR /root/yolov9
RUN wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c-converted.pt

RUN pip install -r requirements.txt
RUN pip3 install opencv-python==4.5.5.64
