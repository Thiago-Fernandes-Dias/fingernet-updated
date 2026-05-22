FROM ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

# Install CUDA 8.0
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget ca-certificates apt-transport-https && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb && \
    wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-core-8-0 \
        cuda-cublas-8-0 \
        cuda-cublas-dev-8-0 \
        cuda-cudart-8-0 \
        cuda-cudart-dev-8-0 \
        cuda-curand-8-0 \
        cuda-curand-dev-8-0 \
        cuda-cusolver-8-0 \
        cuda-cusolver-dev-8-0 \
        cuda-cusparse-8-0 \
        cuda-cusparse-dev-8-0 \
        cuda-nvrtc-8-0 \
        cuda-nvrtc-dev-8-0 \
        cuda-nvml-dev-8-0 \
        cuda-nvgraph-8-0 \
        cuda-nvgraph-dev-8-0 \
        cuda-cufft-8-0 \
        cuda-cufft-dev-8-0 && \
    ln -s cuda-8.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/* /cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

# Install cuDNN 5.1 for CUDA 8.0
RUN wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz && \
    tar -xzf cudnn-8.0-linux-x64-v5.1.tgz && \
    cp cuda/include/cudnn.h /usr/local/cuda/include/ && \
    cp -a cuda/lib64/libcudnn* /usr/local/cuda/lib64/ && \
    ldconfig && \
    rm -rf cuda cudnn-8.0-linux-x64-v5.1.tgz

ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && \
    apt-get install -y python2.7 python2.7-dev python-pip \
    python-numpy python-scipy python-matplotlib \
    libopencv-dev python-opencv \
    graphviz libgraphviz-dev pkg-config git wget unzip && \
    rm -rf /var/lib/apt/lists/*

RUN if [ -e /usr/bin/python ]; then rm /usr/bin/python; fi && ln -s /usr/bin/python2.7 /usr/bin/python

RUN pip install --upgrade pip==20.3.4 setuptools==44.1.1

RUN pip install \
    pydot==1.4.1 \
    tensorflow-gpu==1.0.1 \
    keras==2.0.2 \
    h5py==2.10.0 \
    pyparsing==2.1.4

RUN apt-get update && \
    apt-get install -y --no-install-recommends python-opencv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/FingerNet

COPY ./src /workspace/FingerNet

CMD ["python", "src/train_test_deploy.py", "0", "deploy"]
