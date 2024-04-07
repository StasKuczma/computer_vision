FROM ubuntu:latest

# Install usefull tools
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
    feh
WORKDIR /root/opencv

RUN pip3 install opencv-python==4.5.5.64