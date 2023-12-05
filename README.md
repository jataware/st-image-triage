# image-triage

A Streamlit app for image triage (visualization, exploration, search, annotation)

## User Documentation

__In progress__

## Technical Details

__In progress__

## Installation + Usage

```bash
./build_docker.sh
docker run -P 8501:8501 st-image-triage
```

Navigate to `localhost:8501` in your browser.

_Note:_ Depending on architecture (x86 vs Apple Silicon), `Dockerfile` may need minor adjustments.

## TODO

- [ ] Mouseover to see whole image
- [ ] Customizable filename - current method could have collisions
- [ ] Other file formats than `.zip`