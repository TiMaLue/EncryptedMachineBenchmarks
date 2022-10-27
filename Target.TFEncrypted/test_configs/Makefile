current_dir = $(shell pwd)

all: simpleffnn simpleffnn-pond simpleffnn-securenn mobilenet mobilenet-pond resnet resnet-securenn resnet-pond

mobilenet:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/mobilenet.json test_output/mobilenet.json

mobilenet-pond:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/mobilenet-pond.json test_output/mobilenet-pond.json

mobilenet-securenn:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/mobilenet-securenn.json test_output/mobilenet-securenn.json

resnet:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/resnet.json test_output/resnet.json

resnet-pond:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/resnet-pond.json test_output/resnet-pond.json

resnet-securenn:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/resnet-securenn.json test_output/resnet-securenn.json

simpleffnn:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/simpleffnn.json test_output/simpleffnn.json
simpleffnn-pond:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/simpleffnn-pond.json test_output/simpleffnn-pond.json
simpleffnn-securenn:
	../kill_orphaned_servers.sh
	cd .. && python starter.py test_configs/simpleffnn-securenn.json test_output/simpleffnn-securenn.json

simpleffnn-container:
	touch $(current_dir)/../test_output/simpleffnn-container.json
	docker run -it --rm \
	   -v "$(current_dir)/simpleffnn-container.json":"/runtime_params.json":"ro" \
	   -v "$(current_dir)/../test_output/simpleffnn-container.json":"/runtime_measurements.json" \
	   -v "$(current_dir)/../bench_data/SimpleFFNN/saved-model.pb":"/model":"ro" \
	   -v "$(current_dir)/../bench_data/SimpleFFNN/datasets":"/datasets":"ro" \
	   mpcbenchtarget_tfe /wd/starter.sh /runtime_params.json /runtime_measurements.json
