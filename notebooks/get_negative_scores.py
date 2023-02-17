# %% [markdown]
# # 1.- Load dependencies

# %%
import os
import sys

# add subpath to notebooks
sys.path.append(os.path.abspath(os.path.join('..')))
# sys.path.append(os.path.abspath(os.path.join('../utils')))
sys.path.append('../')

# use first gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations, product
from imutils.paths import list_images
from dense10_segmentator import DenseSegmentator

import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
# dynamically grow GPU memory
config.gpu_options.allow_growth = True
set_session(Session(config=config))

# %% [markdown]
# # 2.- Set path of weights and config file

# %%
# alcohol config
inifile_path = '../config/segmentator_dense10.ini'
modelpath = '../models/weights/dense_models/blurry_gemini_dense_miou_0.9275.h5'
negative_pairs_file = '../ND-LG4000-LR/lists_comparisons/test_non_mated.txt'

model = DenseSegmentator(modelpath=modelpath, inifile_path=inifile_path)
images = sorted(list(list_images('/media/wd_black/datasets/NotreDame-LG4000-LR/iris')))
negative_pairs = []
negative_scores = []

# load indexes from txt file
with open(negative_pairs_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        split = line.split(',')
        imindex1 = int(split[0])
        imindex2 = int(split[1])
        negative_pairs.append((images[imindex1], images[imindex2]))

print('Total images: {}'.format(len(images)))
print('Total negative pairs: {}'.format(len(negative_pairs)))

for imdir1, imdir2 in tqdm(negative_pairs, desc='processing negative pairs'):
    # load images
    image1 = model.load_image(imdir1)
    image2 = model.load_image(imdir2)

    # get iris code
    info1 = model.forward(image1)
    info2 = model.forward(image2)
    code1 = info1.get('iris_code')
    code2 = info2.get('iris_code')
    mask1 = info1.get('mask_iris_rubbersheet')[..., 0]
    mask2 = info2.get('mask_iris_rubbersheet')[..., 0]
    score = model.matchCodes(code1, code2, mask1, mask2)
    negative_scores.append(score)

# %%
# save negative scores to npy file
np.save('negative_scores_indexed.npy', negative_scores)


