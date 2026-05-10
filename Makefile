IMAGE    := fingernet:legacy
GPU      := 0
DATASETS ?= $(PWD)/datasets

.PHONY: build run run_gpu shell shell_gpu clean

build:
	docker build -t $(IMAGE) .

run:
	docker run -it --rm \
		-v "$(PWD)/models":/workspace/FingerNet/models \
		-v "$(DATASETS)":/Datasets \
		$(IMAGE) \
		python train_test_deploy.py 0 deploy

run_gpu:
	docker run -it --rm \
		--gpus all \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		-v "$(PWD)/models":/workspace/FingerNet/models \
		-v "$(DATASETS)":/Datasets \
		$(IMAGE) \
		python train_test_deploy.py $(GPU) deploy

shell:
	docker run -it --rm \
		-v "$(PWD)/models":/workspace/FingerNet/models \
		-v "$(DATASETS)":/Datasets \
		$(IMAGE) \
		bash

shell_gpu:
	docker run -it --rm \
		--gpus all \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		-v "$(PWD)/models":/workspace/FingerNet/models \
		-v "$(DATASETS)":/Datasets \
		$(IMAGE) \
		bash

clean:
	docker rmi $(IMAGE) || true
