#!/usr/bin/env bash
rm -rf mpcbenchrunner
python3 -m venv mpcbenchrunner

source mpcbenchrunner/bin/activate
pip install ../commonsnakes
pip install  -e  ../mpcbenchrunner

