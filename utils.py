import os
import cv2
import random
import numpy as np
from scipy.ndimage.measurements import center_of_mass

def load_image(path, colorspace='RGB'):
    color = colorspace.lower()
    spaces = {
        'rgb': cv2.COLOR_BGR2RGB,
        'hsv': cv2.COLOR_BGR2HSV,
        'hsv_full': cv2.COLOR_BGR2HSV_FULL,
        'gray': cv2.COLOR_BGR2GRAY,
        'lab': cv2.COLOR_BGR2LAB
    }

    if color not in spaces.keys(): 
        print(f'[WARNING] color space {colorspace} not supported')
        print(f'Supported list: {spaces.keys()}')
        print('Colorspace setted to RGB')
        color = 'rgb'
    
    image = cv2.cvtColor(cv2.imread(path), spaces[color])
    return image

def class_mean_iou_old(pred, label):
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    class_iou = []
    
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
        class_iou.append(np.mean(I[index] / U[index]))

    dict_class_iou = {
        'bg': class_iou[0],
        'iris': class_iou[1],
        'pupil': class_iou[2],
        'sclera': class_iou[3],
        'miou': np.mean(class_iou),
        'stdiou:': np.std(class_iou)
    }
    
    return dict_class_iou

def class_mean_iou(pred, label, class_info=None):
    pred = np.argmax(pred, axis=-1)
    label = np.argmax(label, axis=-1)
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)
    
    class_iou = []
    
    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
        class_iou.append(np.mean(I[index] / U[index]))

    if not class_info:
        return np.mean(class_iou), np.std(class_iou)
    else:
        dict_class_iou = dict()
        # class_info: list of dicts
        # [{'source': 'eye', 'id': 0, 'name': 'bg'},
        # {'source': 'eye', 'id': 1, 'name': 'iris'},
        # {'source': 'eye', 'id': 1, 'name': 'pupil'}]
        _ids = [x['id'] for x in class_info]
        _classes = [x['name'] for x in class_info]
        
        if np.all(_ids == unique_labels):
            for lb_id, lb_class in zip(unique_labels, _classes):
                dict_class_iou[lb_class] = class_iou[lb_id]

        else:
            seen_classes = []
            for idx, lb_id in enumerate(unique_labels):
                lb_class = _classes[lb_id]
                seen_classes.append(lb_id)
                dict_class_iou[lb_class] = class_iou[idx]
            
            # not segmented classes cuz there is no class in image. 
            # Append 1.0
            for lb_id in _ids:
                if lb_id in seen_classes: continue
                lb_class = _classes[lb_id]
                dict_class_iou[lb_class] = 0.0
                class_iou.append(0.0)
        
        dict_class_iou['miou'] = np.mean(class_iou)
        dict_class_iou['stdiou'] = np.std(class_iou)
        
        return dict_class_iou