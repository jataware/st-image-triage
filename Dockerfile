FROM python:3.11
# ^ would need to update if you want GPU support

# --
# System-level dependencies

RUN apt clean && apt update -y
RUN apt install -y libvips
RUN pip install --upgrade pip

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ADD . /st-umap
WORKDIR /st-umap
RUN pip install -e .
RUN python scripts/_cache_model.py

# --
# Run

WORKDIR /st-umap
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT [                      \
    "streamlit", "run", "app.py", \
    "--server.port=8501",         \
    "--server.address=0.0.0.0"    \
]
