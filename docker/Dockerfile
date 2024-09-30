FROM tensorflow/tensorflow:2.12.0-gpu-jupyter
LABEL maintainer="jiseok.lee@human.ait.kyushu-u.ac.jp"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt update && apt-get install -y --no-install-recommends git \
    && apt-get install -y python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install scikit-image
RUN pip install tqdm
RUN pip install pydot
RUN pip install seaborn
RUN pip install pandas
RUN pip install stumpy==1.8.0
RUN pip install numba==0.56.0
RUN pip install -U imbalanced-learn
RUN pip install keras==2.9
