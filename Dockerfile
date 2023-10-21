FROM nvcr.io/nvidia/pytorch:22.12-py3
ARG UID=1000
ARG UNAME=testuser
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y openjdk-17-jdk
RUN useradd -u $UID -m $UNAME
USER $UNAME
RUN python --version
RUN python -m pip install --upgrade pip
RUN pip install transformers
RUN pip install ir_datasets ir_measures

ENV IR_DATASETS_HOME="/ir_datasets"
ENV TRANSFORMERS_CACHE="/hf"
ENV PYTERRIER_HOME="/pyterrier"