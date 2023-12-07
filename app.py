#!/usr/bin/env python
"""
    app.py
"""

# --
# Parameters

PORT      = 8501
DEVPORT   = 3001
PERSIST   = True
MODEL_STR = 'openai/clip-vit-base-patch32'
PROD      = True
IMG_DIR   = "images"
OUT_DIR   = "output"
D         = 128
TILESIZE  = 256

# --

import streamlit as st

import os
import pyvips
import shutil
import zipfile
import numpy as np
from glob import glob
from tqdm import tqdm
from stqdm import stqdm
from PIL import Image
from tifffile import imwrite as tiffwrite

import streamlit.components.v1 as components

from ez_feat import featurize, img_safeload, EZDataset

from transformers import CLIPFeatureExtractor, CLIPModel, CLIPProcessor

import layout_funcs as lf

@st.cache_data(show_spinner=False, persist=PERSIST)
def __lf_by_rf(*args, **kwargs):
    return lf.by_rf(*args, **kwargs)

@st.cache_data(show_spinner=False, persist=PERSIST)
def __lf_by_lap(*args, **kwargs):
    return lf.by_lap(*args, **kwargs)

# --
# Setup

st.set_page_config(layout="wide")

if not PROD:
    __st_umap = components.declare_component(
        "st_umap", url=f"http://localhost:{DEVPORT}",
    )
else:
    __st_umap = components.declare_component(
        "st_umap", path="streamlit_umap/frontend/build",
    )

def vip_tiles(inpath, outdir, tilesize, rm=True):
    if os.path.exists(outdir):
        print('vip_tiles: removing old files')
        shutil.rmtree(outdir)
    
    if not os.path.exists(os.path.dirname(outdir)):
        os.makedirs(os.path.dirname(outdir))
    
    _ = (
        pyvips.Image
        .new_from_file(inpath)
        .dzsave(
            outdir,
            overlap    = 0,
            tile_size  = tilesize,
            layout     = "google",
            suffix     = ".png",
            background = [0,0,0]
        )
    )

# --
# Helpers

import torch
@st.cache_resource(show_spinner=False)
class CLIPWrapper:
    def __init__(self, model_str):
        with st.spinner('Loading model...'):
            self.model_str = model_str

            self.tfms      = CLIPFeatureExtractor.from_pretrained(model_str)
            self.processor = CLIPProcessor.from_pretrained(model_str)
            
            model = CLIPModel.from_pretrained(model_str).eval()
            # model = torch.compile(model) # Maybe?
            
            self.vision_model      = model.vision_model
            self.text_model        = model.text_model
            self.visual_projection = model.visual_projection
            self.text_projection   = model.text_projection


@st.cache_data(show_spinner=False, persist=PERSIST)
def _unzip(uploaded_file):
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        mid, *_ = zipfile.Path(zip_ref).iterdir()
        mid = mid.name
    
    if os.path.exists(os.path.join(IMG_DIR, mid)):
        print('_unzip: removing old directory')
        shutil.rmtree(os.path.join(IMG_DIR, mid))
    
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(path=IMG_DIR)

    # TODO - more generic solution
    fnames = sorted(
        glob(os.path.join(f'./{IMG_DIR}', mid, '*.png')) +
        glob(os.path.join(f'./{IMG_DIR}', mid, '*.jpg')) +
        glob(os.path.join(f'./{IMG_DIR}', mid, '*.jpeg'))
    )
    return mid, fnames


@st.cache_data(show_spinner=False, persist=PERSIST)
def _eager_resize(fnames):
    with st.spinner("Resizing images..."):
        for fname in stqdm(fnames):
            _img, _bad = img_safeload(fname, dim=D)
            if _bad: continue
            
            _img = _img.resize((D, D))
            _img.save(fname.replace('.png', f'.{D}.png'))


@st.cache_data(show_spinner=False, persist=PERSIST)
def _compute_embeddings(_model, fnames):
    with st.spinner('Computing features...'):
        embs, bad = featurize(
            _model.vision_model, 
            _model.model_str, 
            EZDataset(fnames, transform=_model.tfms), 
            progress_bar=stqdm,
            # force_device='cpu'
        )
    
    bad    = bad.squeeze()
    embs   = embs[~bad]
    fnames = list(np.array(fnames)[~bad])

    return embs, fnames


def _compute_zooms(outdir):
    minzoom, maxzoom = np.inf, -np.inf
    for xx in os.listdir(outdir):
        try:
            xx      = int(xx)
            minzoom = min((minzoom, xx))
            maxzoom = max((maxzoom, xx))
        except:
            pass
    
    return minzoom, maxzoom


@st.cache_data(show_spinner=False, persist=PERSIST)
def _compute_tiles(mid, df):
    nrow = int(df.row.max() + 1)
    ncol = int(df.col.max() + 1)
    
    # --
    # Write .tif

    with st.spinner('Writing data...'):
      out = np.zeros((D * nrow, D * ncol, 3), dtype=np.uint8)
      for r, c, fname in tqdm(zip(df.row, df.col, df.fname), total=df.shape[0]):
          r, c = int(r), int(c)
          
          _img, _bad = img_safeload(fname.replace('.png', f'.{D}.png'), dim=D)
          if _bad: continue
          
          out[(D*r):(D*r+D), (D*c):(D*c+D)] = np.asarray(_img)
    
      os.makedirs(os.path.join(OUT_DIR, mid), exist_ok=True)
      tiffwrite(os.path.join(OUT_DIR, mid, 'grid.tif'), out)

    # --
    # saves the tiles
    
    with st.spinner('Computing tiles...'):
        # !! BUG - remove this dependency
        vip_tiles(
            os.path.join(OUT_DIR, mid, 'grid.tif'), 
            outdir=os.path.join('static/tiles/', mid),
            tilesize=TILESIZE
        )


def make_config(mid, df):
    nrow = int(df.row.max() + 1)
    ncol = int(df.col.max() + 1)

    fnames_layout    = np.zeros((nrow, ncol), dtype=object)
    fnames_layout[:] = None
    for r, c, fname in zip(df.row, df.col, df.fname):
        fnames_layout[r, c] = fname
    
    minzoom, maxzoom = _compute_zooms(os.path.join('static/tiles/', mid))
    
    tile_url = f"/app/static/tiles/{mid}/{{z}}/{{y}}/{{x}}.png"
    if not PROD:
        tile_url = f"http://localhost:{PORT}" + tile_url
    
    f2coord = {f:(nrow - r - 1, c) for r, c, f in df[['row', 'col', 'fname']].values}
    
    return {
        "mid"      : mid,
        "minZoom"  : minzoom,
        "maxZoom"  : maxzoom,
        "tileSize" : TILESIZE,
        "nrow"     : nrow,
        "ncol"     : ncol,
        "name"     : mid,
        
        "fnames"   : list(fnames_layout[::-1].ravel()), # CAREFUL!
        
        "classes"  : st.session_state.classes,
        "f2label"  : st.session_state.f2label,
        "f2coord"  : f2coord,
        
        "tiles"    : [tile_url]
    }

# --
# Run

def main():

    # --
    # Sidebar
    
    with st.sidebar:
        st.markdown("## Image Triage")

        with st.expander("Documentation"):
                st.markdown("""
                    1) Upload a .zip file containing a directory of images.
                    2) The app will use OpenAI's CLIP neural network to embed the images.
                    3) There are a number of ways to compute the layout of the data:
                        
                        a) (Default) Explore - Layout images such that "similar" images are close together
                        
                        b) Text Query - Find images that are "similar" to an English text query
                        
                        c) Image Query - Find images that are "similar" to an uploaded image
                        
                        d) Query by Class - Find images that are "similar" to an annotation class
                    
                    Note that in {a, b, c}, images are orders left-to-right, top-to-bottom, so the "most similar" images are in the top left corner.
                    
                    4) Annotate data by 
                        - adding a class name with the "Add Classes..." menu
                        - hovering over images in the display while holding the `1` key for the first class, `2` key for the second class, etc
                        - to save annotations, press the `command` key 
                    
                    5) download for downstream use.
                    
                    _Note:_ This is beta software, and we'd love to hear about bugs / feedback.  _Please_ contact Data Team / @bkj on wildfire for help!
                """)
        
        uploaded_file = st.file_uploader("Upload Images", type='zip')
        if uploaded_file is None: return
        
        st.divider()

        # --
        # Generic prep
        
        clip         = CLIPWrapper(MODEL_STR)
        _mid, fnames = _unzip(uploaded_file)
        embs, fnames = _compute_embeddings(clip, fnames)
        _eager_resize(fnames)
        
        if 'buff' not in st.session_state:
            st.session_state['buff'] = None
        if 'f2label' not in st.session_state:
            st.session_state['f2label'] = {f:None for f in fnames}
        if 'classes' not in st.session_state:
            st.session_state['classes'] = [None]
        
        # --
        # Layout
        
        def roll_buff(*args, **kwargs):
            if st.session_state.buff is not None:
                st.session_state.f2label, st.session_state.classes = st.session_state.buff
                st.session_state.buff = None
        
        query_mode = st.radio(
            "Query Mode", 
            options=[
                "Explore (Unsupervised)",
                "Text Query",
                "Image Query",
                "Query By Class"
            ],
            on_change=roll_buff
        )
        
        df = None
        with st.spinner("Computing layout..."):
            print(query_mode)
            if query_mode == "Explore (Unsupervised)":
                # df = _compute_layout_rf(fnames, embs)
                mid = f'{_mid}-lap'
                df  = __lf_by_lap(fnames, embs)
            
            elif query_mode == "Text Query":
                query = st.text_input("Query", on_change=roll_buff)
                if not query: return
                print(f'query={query}')
                mid = f'{_mid}-{query}'
                df  = lf.by_text(clip, query, fnames, embs)            
            
            elif query_mode == "Image Query":
                query_file = st.file_uploader("Upload Query Images", type=['jpg', 'png'])
                if query_file is None: return
                mid = f'{_mid}-{query_file.file_id}'
                df  = lf.by_img(clip, Image.open(query_file), fnames, embs)
                
            elif query_mode == "Query By Class":
                f2label = st.session_state.f2label
                classes = [xx for xx in st.session_state.classes if xx is not None]
                if len(classes) == 0:
                    st.text("!! No classes yet...")
                else:
                    target_class = st.radio(
                        "Target Class", 
                        options=classes,
                        on_change=roll_buff
                    )
                    
                    mid = f'{_mid}-qbc-{target_class}' # !! collisions?
                    df  = lf.by_clf(target_class, fnames, embs, f2label)
            else:
                raise Exception('!!')
        
        df['lab'] = df.fname.apply(st.session_state.f2label.get)
        _compute_tiles(mid, df[['fname', 'row', 'col']])

    # --
    # Main panel
    
    config = make_config(mid, df)
    res    = __st_umap(config=config)
    if res:
        f2label, _       = res
        df['lab']        = df.fname.apply(f2label.get)
        st.session_state['buff'] = res
    
    st.divider()
    
    # --
    # Download Buttons
    
    df_out = df.copy()
    df_out.fname = df_out.fname.apply(lambda x: x.replace(f'./{IMG_DIR}/', ''))
    df_out = df_out.rename(columns={"fname" : "img_path"})
    df_out = df_out[['img_path', 'lab']]
    
    _, col1, _, _ = st.columns([1,1,1,1])

    with col1:
        st.download_button('Download Annotations', df_out.to_csv(index=False), f'{config["mid"]}_annotations.csv')

    # !! BUG: Need to make sure filenames are sorted/matched properly
    
    # --
    # Display annotations

    st.dataframe(df_out, use_container_width=True, hide_index=True)
    print('!!!! DONE')

if __name__ == "__main__":
    main()