FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG USER=docker_nano
ARG UID=1000
ARG GID=1000
ARG PW=user
RUN id

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 gcc g++ -y

RUN python3 -m pip install wandb black

# Create the user
RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}
RUN chown -R ${USER}:${USER} /home/${USER}
USER ${UID}:${GID}
