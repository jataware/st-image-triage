# image-triage

A Streamlit app for image triage (visualization, exploration, search, annotation)

## Installation + Usage

```bash
docker build -t st-image-triage .
docker run -p 8501:8501 st-image-triage
```

Navigate to `localhost:8501` in your browser. UI usage documented in app.

_Note:_ Tested on Ubuntu 22.04 / CPU-only on 2023-12-06.

## Development

### Modifying .js

Install NPM:
```bash
sudo apt update
sudo apt install nodejs npm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
nvm install stable
```

Build code:
```bash
cd streamlit_umap/frontend/build
npm install
npm run build
cd ../../../
```
