#!/bin/bash

source "./venv/bin/activate"

export HF_HOME
export HF_ENDPOINT
export PIP_INDEX_URL
export PYTHONUTF8

python gui.py "$@"


