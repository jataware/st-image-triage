#!/bin/bash

# run.sh

# --
# Create env

conda create -y -n stit_env python=3.11
conda activate stit_env
pip install -r osx-requirements.txt