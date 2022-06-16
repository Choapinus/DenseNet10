import abc
import cv2
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("Segmentator")

class AbstractSegmentatorClass(metaclass=abc.ABCMeta):
    """Dense segmentator abstract base class."""

    def __init__(
        self, *args, modelpath='', threshold=0., pooling_operation='max', **kwargs
    ):
        """Class instantiation.

        Parameters
        ----------
        modelpath : str
            Path to hdf5 model
        threshold : float, optional
            Mask threshold. None to use argmax or set 
            a threshold greater than 0.0 (0.7 recommended)
        """
        self.threshold = threshold
        self.modelpath = modelpath
        self.pooling_operation = pooling_operation.lower()
        self.model = self.set_model(*args, **kwargs)
        self.target_size = self.get_target_size(*args, **kwargs)
        
        logger.debug(f"{type(self).__name__} instanciated\n"
                     f"Model path: '{modelpath}'\nTarget size: {self.target_size}\n"
                     f"Mask threshold: {threshold}")
    
    @abc.abstractmethod
    def set_model(self, *args, **kwargs):
        raise NotImplementedError('set_model method must be implemented')
    
    @abc.abstractmethod
    def set_feature_extractor(self, *args, **kwargs):
        raise NotImplementedError('set_feature_extractor method must be implemented')
    
    @abc.abstractmethod
    def get_target_size(self, *args, **kwargs):
        raise NotImplementedError('get_target_size method must be implemented')
    
    @abc.abstractmethod
    def eval_generator(self, *args, **kwargs):
        raise NotImplementedError('eval_generator method must be implemented')
    
    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError('predict method must be implemented')
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward method must be implemented')
    
    def summary(self):
        return self.model.summary()
    
    def count_params(self):
        return self.model.count_params()
    
    def load_image(self, path, colorspace='RGB'):
        color = colorspace.lower()

        spaces = {
            'rgb': cv2.COLOR_BGR2RGB,
            'hsv': cv2.COLOR_BGR2HSV,
            'hsv_full': cv2.COLOR_BGR2HSV_FULL,
            'gray': cv2.COLOR_BGR2GRAY,
            'lab': cv2.COLOR_BGR2LAB
        }

        if color not in spaces.keys(): 
            logger.warn(f"Color space {colorspace} not supported")
            logger.warn(f"Supported list: {spaces.keys()}")
            logger.warn("Colorspace setted to RGB")
            color = 'rgb'
        
        image = cv2.cvtColor(cv2.imread(path), spaces[color])

        return image
    
    def eval_iou(self, pred, label, class_info=None, return_bg=False):
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
            if return_bg:
                return np.mean(class_iou), np.std(class_iou)
            else:
                return np.mean(class_iou[1:]), np.std(class_iou[1:])
        else:
            dict_class_iou = dict()
            # class_info: list of dicts
            # [{'source': 'eye', 'id': 0, 'name': 'bg'},
            # {'source': 'eye', 'id': 1, 'name': 'iris'},
            # {'source': 'eye', 'id': 2, 'name': 'pupil'}]
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
                # Append 0.0
                for lb_id in _ids:
                    if lb_id in seen_classes: continue
                    lb_class = _classes[lb_id]
                    dict_class_iou[lb_class] = 0.0
                    class_iou.append(0.0)
            
            if return_bg:
                dict_class_iou['miou'] = np.mean(class_iou)
                dict_class_iou['stdiou'] = np.std(class_iou)
            else:
                dict_class_iou['miou'] = np.mean(class_iou[1:])
                dict_class_iou['stdiou'] = np.std(class_iou[1:])
                # borrar bg
        
        return dict_class_iou
    
    def process_image(self, image=None, preprocessing=[], tsize=None):
        """Resample and process the input image. The image is first
        resampled using OpenCV with cv2.INTER_AREA interpolation for
        decimation, or cv2.INTER_CUBIC for expansion.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        preprocessing : function, optional
            Any number of preprocessing functions as positional
            arguments. They can also be contained in a sequence or
            collection, but must be unpacked in the function call. Each
            function must receive a NumPy array as argument, and output
            a NumPy array of the same rank.

        Returns
        -------
        numpy.ndarray
            Reshaped and processed image.
        tuple
            Original size of input image.
        """

        original_shape = image.shape[:-1]

        if tsize == None: tsize = self.target_size
        
        if image.shape[:2] > tsize[::-1]:
            image = cv2.resize(
                image, tsize,
                interpolation=cv2.INTER_AREA
            )
        elif image.shape[:2] < tsize[::-1]:
            image = cv2.resize(
                image, tsize,
                interpolation=cv2.INTER_CUBIC
            )

        for f in preprocessing:
            image = f(image)

        # normalize image
        image = image.astype(np.float32) / 255.

        return image, original_shape
    
    def bitwise_image(self, image, mask):
        """Do bitwise operation to extract only image pixels
        given a mask.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.
        mask : numpy.ndarray
            Predicted mask with [height, width] format.

        Returns
        -------
        numpy.ndarray
            Bitwised image with [height, width, channels]
        """

        mask = cv2.merge((mask, mask, mask))
        mask = (mask*255).astype(np.uint8)
        image = mask & image # np.logical_and

        return image.astype(np.uint8)