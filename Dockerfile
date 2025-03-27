FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN \
        apt-get update && \
        apt-get install -y git cmake vim wget curl sudo python3.10 pip tmux htop

# Install miniconda
RUN \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    chmod +x /opt/conda/etc/profile.d/conda.sh && \
    echo "/opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda init" >> ~/.bashrc

SHELL \
        ["/bin/bash", "-l", "-c"]