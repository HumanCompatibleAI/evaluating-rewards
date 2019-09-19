#!/bin/bash

PATH_TO_VENV=$1

virtualenv -p python3.7 ${PATH_TO_VENV}
source ${PATH_TO_VENV}/bin/activate
pip install -r requirements.txt