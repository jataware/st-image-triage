"""
    layout_funcs.py
"""

import numpy as np
import pandas as pd
from umap import UMAP
import rasterfairy as rf
from sklearn.svm import LinearSVC
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import torch

def to_numpy(x):
    return x.detach().cpu().numpy()

# --
# Unsupervised

# @st.cache_data(show_spinner=False, persist=PERSIST)
def by_rf(fnames, embs):
    layout = UMAP(n_neighbors=5, min_dist=0.1, metric='cosine', verbose=True).fit_transform(embs)

    grid_cr, (_, _) = rf.transformPointCloud2D(layout)
    grid_cr = grid_cr.astype(int)

    df        = pd.DataFrame({'fname' : fnames})
    df['col'] = grid_cr[:,0]
    df['row'] = grid_cr[:,1]
    # df        = df.sort_values(["row", "col"]).reset_index(drop=True)
    
    return df

# @st.cache_data(show_spinner=False, persist=PERSIST)
def by_lap(fnames, embs):
    print('by_lap: start')
    layout = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', verbose=True).fit_transform(embs)
    
    n_pts  = len(fnames)
    n_gpts = 2 * np.ceil(np.sqrt(n_pts))
    r, c = np.meshgrid(
        np.arange(n_gpts),
        np.arange(n_gpts),
    )
    gpts = np.column_stack([r.ravel(), c.ravel()])

    layout -= layout.min(axis=0, keepdims=True)
    layout /= layout.max(axis=0, keepdims=True)
    layout *= n_gpts

    cost = cdist(layout, gpts)

    _, c_idx = linear_sum_assignment(cost)
        
    df        = pd.DataFrame({'fname' : fnames})
    df['col'] = gpts[c_idx][:,0].astype(int)
    df['row'] = gpts[c_idx][:,1].astype(int)
    # df        = df.sort_values(["row", "col"]).reset_index(drop=True)
    
    return df

# --

def _layout_by_sim(fnames, sim):
    grid_d = int(np.ceil(np.sqrt(len(sim))))
    topk   = np.argsort(-sim)
    row    = np.array([i // grid_d for i in range(len(sim))])
    col    = np.array([i % grid_d for i in range(len(sim))])

    df        = pd.DataFrame({'fname' : fnames})
    df['row'] = row[np.argsort(topk)]
    df['col'] = col[np.argsort(topk)]
    # df        = df.sort_values(["row", "col"]).reset_index(drop=True)

    return df

def by_text(clip, query, fnames, embs):
    _ = clip.text_model.to('cpu')
    
    with torch.inference_mode():
        # prep text embeddings
        qenc = clip.text_model(**clip.processor(text=[query], return_tensors='pt'))
        qenc = clip.text_projection(qenc.pooler_output)
        qenc = to_numpy(qenc).squeeze()
        qenc = qenc / np.sqrt((qenc ** 2).sum(axis=-1, keepdims=True))

        # prep visual embeddings
        venc = embs @ clip.visual_projection.weight.numpy().T # !! annoying
        venc = venc / np.sqrt((venc ** 2).sum(axis=-1, keepdims=True))

    return _layout_by_sim(fnames, sim=venc @ qenc)


def by_img(clip, query, fnames, embs):
    _ = clip.vision_model.to('cpu')

    with torch.inference_mode():
        # query embeddings
        qenc = clip.vision_model(**clip.processor(images=[query], return_tensors='pt'))
        # <<
        # qenc = clip.vision_projection(qenc.pooler_output) # !! BUG - Am I supposed to use this?
        # --
        qenc = qenc.pooler_output
        # >>
        qenc = to_numpy(qenc).squeeze()
        qenc = qenc / np.sqrt((qenc ** 2).sum(axis=-1, keepdims=True))

        # prep visual embeddings
        # <<
        # venc = embs @ clip.visual_projection.weight.numpy().T
        # --
        venc = embs
        # >>
        venc = venc / np.sqrt((venc ** 2).sum(axis=-1, keepdims=True))

    return _layout_by_sim(fnames, sim=venc @ qenc)

def by_clf(target_class, fnames, embs, f2label):
    y   = [v == target_class for v in f2label.values()]
    clf = LinearSVC().fit(embs, y)
    sim = clf.decision_function(embs)

    return _layout_by_sim(fnames, sim)