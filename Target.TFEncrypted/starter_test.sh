#!/usr/bin/env bash

#docker run -it --rm \
#    -v "$(pwd)/starter_test_params.json":"/runtime_params.json":"ro" \
#    -v "$(pwd)/starter_test_output.json":"/runtime_measurements.json" \
#    -v "$(pwd)/bench_data/SimpleLogisticReg/model.onnx":"/model":"ro" \
#    -v "$(pwd)/bench_data/SimpleLogisticReg/datasets":"/datasets":"ro" \
#    mpcbenchtarget_tfe /wd/starter.sh /runtime_params.json /runtime_measurements.json
#

docker run -it --rm \
    -v "$(pwd)/starter_test_params.json":"/runtime_params.json":"ro" \
    -v "$(pwd)/starter_test_output.json":"/runtime_measurements.json" \
    -v "$(pwd)/bench_data/SimpleFFNN/model.onnx":"/model":"ro" \
    -v "$(pwd)/bench_data/SimpleFFNN/datasets":"/datasets":"ro" \
    -v "$(pwd)/starter.py":"/wd/starter.py":"ro" \
    mpcbenchtarget_tfe bash
