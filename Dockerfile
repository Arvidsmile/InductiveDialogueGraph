# Tensorflow needs to be be 1.3.0 or we get 'module has no attribute 'app' errors
FROM python
FROM tensorflow/tensorflow:1.3.0

LABEL maintainer="Arvid.Lindstrom@gmail.com"

#RUN pip install networkx==1.11
RUN pip install networkx==1.11

RUN apt-get update
RUN apt-get install nano

# This is added to make sure we have write-permission
# to any data created inside the docker-container
ARG USER_ID

RUN adduser --disabled-password --gecos '' --uid $USER_ID user
USER user

WORKDIR /InductiveDialogueGraph