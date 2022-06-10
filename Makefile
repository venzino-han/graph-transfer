
default: build

help:
	@echo 'Management commands for graph-transfer:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build image'
	@echo '    make run              Start docker container'
	@echo '    make up               Build and run'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t graph-transfer 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus 0 --ipc=host --name graph-transfer -v `pwd`:/workspace/graph-transfer graph-transfer:latest /bin/bash

up: build run

rm: 
	@docker rm graph-transfer

stop:
	@docker stop graph-transfer

reset: stop rm