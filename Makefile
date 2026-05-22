IMAGE    := fingernet:latest
GPU      := 0
DATASETS ?= $(PWD)/datasets
OUTPUT_DIR ?= $(PWD)/output

.PHONY: build run run_gpu shell shell_gpu clean

build:
	docker build -t $(IMAGE) .

run:
	docker run --rm \
		-v "$(PWD)/models":/Models \
		-v "$(DATASETS)":/Datasets \
		-v "$(OUTPUT_DIR)":/Output \
		$(IMAGE) \
		python train_test_deploy.py 0 deploy

run_gpu:
	docker run --rm \
		--gpus all \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		-v "$(PWD)/models":/Models \
		-v "$(DATASETS)":/Datasets \
		-v "$(OUTPUT_DIR)":/Output \
		$(IMAGE) \
		python train_test_deploy.py $(GPU) deploy

shell:
	docker run -it --rm \
		-v "$(PWD)/models":/Models \
		-v "$(DATASETS)":/Datasets \
		-v "$(OUTPUT_DIR)":/Output \
		$(IMAGE) \
		bash

shell_gpu:
	docker run -it --rm \
		--gpus all \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		-v "$(PWD)/models":/Models \
		-v "$(DATASETS)":/Datasets \
		-v "$(OUTPUT_DIR)":/Output \
		$(IMAGE) \
		bash

clean:
	docker rmi $(IMAGE) || true
