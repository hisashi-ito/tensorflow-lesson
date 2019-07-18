# GPU
FROM tensorflow/tensorflow:1.8.0-gpu-py3
LABEL maintainer="username"

RUN apt-get -y update && apt-get install -y --no-install-recommends wget vim git

RUN apt install graphviz -y
RUN apt-get install -y language-pack-ja-base language-pack-ja
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
RUN echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y time
RUN pip3 --no-cache-dir install tensorflow-gpu==1.11.0 tensorboard==1.11.0 numpy==1.15.2
