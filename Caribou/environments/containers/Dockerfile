# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster
WORKDIR /home/nicolas/github/Caribou/
COPY requirements.txt requirements.txt
RUN python3 -m venv .
RUN source ./bin/activate
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .

# docker build -f /home/nicolas/github/Caribou/Caribou/environments/containers/Dockerfile .
