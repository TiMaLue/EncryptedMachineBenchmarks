
all: mpcbenchtarget_crypten

mpcbenchtarget_crypten: Dockerfile
	docker build -f ./Dockerfile -t mpcbenchtarget_crypten .

gauss_data:
	python3 -m benchmark.gauss_dataset
	rm -rf bench_data/SimpleFFNN/datasets
	mkdir -p bench_data/SimpleFFNN/datasets
	mkdir -p bench_data/SimpleLogisticReg/datasets
	cp -r datasets/gauss_clusters_20_1/* bench_data/SimpleFFNN/datasets
	cp -r datasets/gauss_clusters_2_1/* bench_data/SimpleLogisticReg/datasets

gauss_models:
	python3 -m benchmark.gauss_models train store_nn load_nn test
	rm -rf bench_data/SimpleFFNN/model.onnx
	rm -rf bench_data/LogisticReg/model.onnx
	mkdir -p bench_data/SimpleFFNN
	mkdir -p bench_data/SimpleLogisticReg
	cp -r pre_trained_models/feedforward_neural_network.onnx bench_data/SimpleFFNN/model.onnx
	cp -r pre_trained_models/logistic_regression.onnx bench_data/SimpleLogisticReg/model.onnx

vision_models:
	mkdir -p bench_data/image_cls
	mkdir -p bench_data/image_cls_quant
	rsync --update pre_trained_models/image_cls/*.onnx bench_data/image_cls
	rsync --update pre_trained_models/image_cls_quant/*.onnx bench_data/image_cls_quant

vision_data:
	rm -rf bench_data/image_cls/datasets
	mkdir -p bench_data/image_cls/datasets
	cp -r datasets/image_cls/* bench_data/image_cls/datasets

