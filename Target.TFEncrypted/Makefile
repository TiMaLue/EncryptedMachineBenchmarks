
all: mpcbenchtarget_tfe

mpcbenchtarget_tfe: Dockerfile
	docker build -f ./Dockerfile -t mpcbenchtarget_tfe .

simple_models:
	python ./conv_model.py ./bench_data/SimpleFFNN/model.onnx ./bench_data/SimpleFFNN/saved-model.pb
	python ./conv_model.py ./bench_data/SimpleLogisticReg/model.onnx ./bench_data/SimpleLogisticReg/saved-model.pb

img_cls_models:
	python ./tf_model_download.py ./bench_data
