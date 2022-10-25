bs = 6
dim = (240, 320)
# dim = (224, 224)
fix_cuda = True

import os
import cv2
import json
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import keras
from metrics import mean_iou, mean_dice
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from utils.datagenerator import OpenEDS, EyeDataset
from keras_fc_densenet import _create_fc_dense_net, build_FC_DenseNet10
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from imgaug import (
    augmenters as iaa,
)  # https://github.com/aleju/imgaug (pip3 install imgaug)
from augmentations import avg_aug

from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import ConfigProto

if fix_cuda:
    config = ConfigProto()
    # dynamically grow GPU memory because unexpected error with gtx 1660 Ti max-Q
    config.gpu_options.allow_growth = True
    set_session(Session(config=config))


# dataset contains pupil, iris, sclera and eye marks
# dataset_dir = "/home/choppy/TOC/datasets/alcohol/alcohol_v7.0"
dataset_dir = "/home/choppy/TOC/datasets/alcohol/alcohol_v4"

print("[INFO] Loading dataset")

# transform those marks into a tensor with a custom generator and avg_aug
# trainG = OpenEDS(batch_size=bs, augmentation=avg_aug(), dim=dim)
# trainG = EyeDataset(batch_size=bs, augmentation=avg_aug(), dim=dim)
trainG = EyeDataset(batch_size=bs, augmentation=None, dim=dim)
trainG.load_eyes(dataset_dir, "train")
trainG.prepare()

# valG = OpenEDS(batch_size=bs, dim=dim)
valG = EyeDataset(batch_size=bs, dim=dim)
valG.load_eyes(dataset_dir, "val")
valG.prepare()

print("[INFO] End loading")

# store models
os.makedirs("models", exist_ok=True)

densenet_params = {
    "img_input": trainG.input_shape,
    "nb_classes": trainG.num_classes,
    # avg params of dense10
    "nb_dense_block": 4,
    "growth_rate": 6,
    "nb_layers_per_block": 3,
    "init_conv_filters": 48,
    "transition_pooling": "max",
    "initial_kernel_size": (3, 3),
    "upsampling_type": "deconv",
    "activation": "relu",
    "bn_momentum": 0.9,
    "weight_decay": 1e-4,
    "dropout_rate": 0.15,
    "final_softmax": True,
    "name_scope": "DenseNetFCN",
    "data_format": "channels_last",
}

# save config cuz "no memory" is left
with open("models/config.json", "w") as fp:
    json.dump(densenet_params, fp)

# new model
model = build_FC_DenseNet10(
    nb_classes=trainG.num_classes,
    final_softmax=densenet_params.get("final_softmax"),
    input_shape=densenet_params.get("img_input"),
    dropout_rate=densenet_params.get("dropout_rate"),
    data_format=densenet_params.get("data_format"),
)

# compile model with adam optimizer
# experimental: with rmprop optimizer
model.compile(
    # optimizer=Adam(lr=1e-4, decay=1e-7),
    optimizer=RMSprop(learning_rate=1e-4, decay=1e-6, momentum=0.9, centered=False),
    # optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=[
        mean_iou,
        mean_dice,
    ],
)

# save checkpoints of model
mckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/epoch_{epoch:03d}_miou_{val_mean_iou:.4f}.h5",
    monitor="val_mean_iou",
    verbose=1,
    save_best_only=True,
    mode="max",
)

# save tensorboard models
tsboard = tf.keras.callbacks.TensorBoard(
    write_images=True,
    write_graph=True,
)

# reduce lr on plateau and reduce with a factor of 0.5
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_mean_iou", factor=0.5, patience=10, verbose=1, mode="max", min_lr=1e-15
)

# early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_mean_iou",
    patience=30,
    mode="max",
    # restore_best_weights=True,
)


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    # supress deprecated warnings of imgcorruptlike: "multichannel is a deprecated argument of gaussian"
    warnings.simplefilter("ignore")
    fxn()
    # fit generator into model and recover history
    hist = model.fit_generator(
        trainG,
        validation_data=valG,
        epochs=300,
        verbose=1,
        steps_per_epoch=len(trainG),
        validation_steps=len(valG),
        callbacks=[reduce_lr, mckpt, tsboard, early_stop],
        # callbacks=[reduce_lr, mckpt, tsboard, ],
        workers=12,
        max_queue_size=36,
        use_multiprocessing=True,  # use wisely
    )

# define a plot function to plot history scores
def plot_history(history, title, save_path, figsize=(10, 8), font_scale=2, linewidth=4):
    with sns.plotting_context(
        "notebook", font_scale=2, rc={"lines.linewidth": linewidth}
    ):
        epochs = len(history.history["val_loss"])
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_ylabel("Loss/IoU.")
        ax.set_xlabel("Epochs")
        sns.lineplot(range(epochs), history.history["loss"], label="Train Loss", ax=ax)
        sns.lineplot(
            range(epochs), history.history["val_loss"], label="Val. Loss", ax=ax
        )
        sns.lineplot(
            range(epochs), history.history["mean_iou"], label="Train mIoU.", ax=ax
        )
        sns.lineplot(
            range(epochs), history.history["val_mean_iou"], label="Val. mIoU.", ax=ax
        )
        fig.savefig(save_path)


# call plot function
plot_history(hist, "Dense10 Training Loss & mIoU.", "models/train_plot.png")
