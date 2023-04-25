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
import argparse
import numpy as np
import pandas as pd
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

# make arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default='../models/weights/dense_models/blurry_gemini_dense_miou_0.9275.h5')
parser.add_argument('--inifile_path', type=str, default='../config/segmentator_dense10.ini')
parser.add_argument('--folder', type=str, default='/media/wd_black/datasets/NotreDame-LG4000-LR/iris')
parser.add_argument('--store_masks', action='store_true', default=False) # store masks in folder
parser.add_argument('--store_codes', action='store_true', default=False) # store iris codes in folder
parser.add_argument('--store_radii', action='store_true', default=False) # store radii in csv file
parser.add_argument('--store_rubbersheet', action='store_true', default=False) # store images in folder

args = parser.parse_args()

images = sorted(list(list_images(args.folder)))

# make folders to store masks, codes and radii
folder_name = args.folder.split(os.sep)[-1]
folder_name = 'dense10-results/' + folder_name

if args.store_masks:
    os.makedirs(os.path.join(folder_name, 'masks'), exist_ok=True)
if args.store_codes:
    os.makedirs(os.path.join(folder_name, 'codes'), exist_ok=True)
if args.store_radii:
    os.makedirs(os.path.join(folder_name, 'radii'), exist_ok=True)
    # make a df to store radii
    radii_df = pd.DataFrame(columns=['iris_x', 'iris_y', 'iris_r', 'pupil_x', 'pupil_y', 'pupil_r'])
if args.store_rubbersheet:
    os.makedirs(os.path.join(folder_name, 'rubbersheet'), exist_ok=True)

# create model
model = DenseSegmentator(modelpath=args.modelpath, inifile_path=args.inifile_path)

# for each image in folder do forward pass and store results
for imdir in tqdm(images, desc='processing images'):
    # load image
    image = model.load_image(imdir)

    try:
        # forward pass
        info = model.forward(image)
        
        # get iris code and mask
        masks = info.get('mask')
        code = info.get('iris_code')
        rubber = info.get('iris_rubbersheet')
        mask_rubber = info.get('mask_iris_rubbersheet')

        # get iris and pupil radii
        iris_x, iris_y, iris_r = info.get('iris_x'), info.get('iris_y'), info.get('iris_r_max')
        pupil_x, pupil_y, pupil_r = info.get('pupil_x'), info.get('pupil_y'), info.get('pupil_r_max')

        # store results
        if args.store_masks:
            pass
        if args.store_codes:
            pass
        if args.store_radii:
            # guardar asi los radios [iriscenterX, iriscenterY, irisradius, pupilcenterX, pupilcenterY, pupilradius]
            radii_df = radii_df.append(
                {
                    'iris_x': iris_x, 'iris_y': iris_y, 'iris_r': iris_r, 
                    'pupil_x': pupil_x, 'pupil_y': pupil_y, 'pupil_r': pupil_r
                }, ignore_index=True
            )
        if args.store_rubbersheet:
            pass



    except Exception as ie:
        # print readable info about error, image and its info stored in info
        print('Error processing images: {} and {}'.format(imdir))
        print('Check processed information like iris and pupil radii')
        continue

# try catch were applied to avoid errors when processing images
# those errors are due to the fact that some radius of pupil and 
# iris are too high or not detected at all by the ***radii estimator function***

# store radii in csv file
if args.store_radii:
    # define filename using radii estimator function
    filename = f'{model.rtype}_radii.csv'
    radii_df.to_csv(os.path.join(folder_name, 'radii', filename), index=False)