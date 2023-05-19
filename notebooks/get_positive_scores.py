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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
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
positive_pairs_file = '../ND-LG4000-LR/lists_comparisons/test_mated.txt'
nd_lg4000_path = '/media/wd_black/datasets/NotreDame-LG4000-LR/iris'
images = sorted(list(list_images(nd_lg4000_path)))
model = DenseSegmentator(modelpath=modelpath, inifile_path=inifile_path)
positive_pairs = []
positive_scores = []

# load indexes from txt file
with open(positive_pairs_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        split = line.split(',')
        imindex1 = int(split[0])
        imindex2 = int(split[1])
        positive_pairs.append((images[imindex1], images[imindex2]))

print('Total images: {}'.format(len(images)))
print('Total positive pairs: {}'.format(len(positive_pairs)))

# %%


for imdir1, imdir2 in tqdm(positive_pairs, desc='processing positive pairs'):
    # load images
    image1 = model.load_image(imdir1)
    image2 = model.load_image(imdir2)

    try:
        # get iris code
        info1 = model.forward(image1)
        info2 = model.forward(image2)
    except Exception as ie:
        # print readable info about error, image and its info stored in info1 and info2
        print('Error processing images: {} and {}'.format(imdir1, imdir2))
        print('Check processed information like iris and pupil radii')
        continue

    code1 = info1.get('iris_code')
    code2 = info2.get('iris_code')
    mask1 = info1.get('mask_iris_rubbersheet')[..., 0]
    mask2 = info2.get('mask_iris_rubbersheet')[..., 0]
    
    try:
        score = model.matchCodes(code1, code2, mask1, mask2)
    except Exception as e:
        print('Error matching codes: {} and {}'.format(imdir1, imdir2))
        print('Check processed information like iris and pupil radii')
        continue
    
    positive_scores.append(score)

# save positive scores to npy file
np.save(f'positive_scores_{model.rtype}.npy', positive_scores)

# try catch were applied to avoid errors when processing images
# those errors are due to the fact that some radius of pupil and 
# iris are too high or not detected at all by the ***radii estimator function***

# COM: no errors
# LMS2: a lot of images threw errors
# LMS3: no errors

# agregar txt con pasos para correr el codigo