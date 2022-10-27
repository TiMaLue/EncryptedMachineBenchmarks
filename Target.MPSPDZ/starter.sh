#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHONUNBUFFERED=1 python3 "$SCRIPT_DIR"/starter.py "$@"
