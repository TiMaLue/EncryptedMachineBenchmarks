#!/bin/bash
echo on
docker run -it --user 1000:1000 --rm -p 8888:8888 -v $(pwd):/wd mpcbench_viz jupyter notebook --ip="0.0.0.0" --NotebookApp.token='' --NotebookApp.password=''
