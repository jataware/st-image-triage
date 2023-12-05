"""
    _download_model.py
    
    Load a model to force caching
"""

import argparse
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_str', default='openai/clip-vit-base-patch32')
    args  = parser.parse_args()
    return args

args = parse_args()

print(f'Caching {args.model_str}')
model_str    = args.model_str
tfms         = CLIPFeatureExtractor.from_pretrained(model_str)
processor    = CLIPProcessor.from_pretrained(model_str)
model        = CLIPModel.from_pretrained(model_str).eval()
print('\t Done!')