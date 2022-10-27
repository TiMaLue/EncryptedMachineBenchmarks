#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHONUNBUFFERED=1 conda run -n main python3 "$SCRIPT_DIR"/starter.py "$@"

