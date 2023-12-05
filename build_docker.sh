#!/bin/bash

# build_docker.sh

cd streamlit_umap/frontend
bun install
bun run build
rm -rf node_modules
cd ../../

docker build -t st-image-triage .