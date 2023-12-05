#!/bin/bash

# clean.sh

echo "clearing streamlit cache"
streamlit cache clear

echo "removing ./output"
rm -rf output 

echo "removing ./images"
rm -rf images 

echo "removing ./static"
rm -rf static