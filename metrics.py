import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from scipy.ndimage.measurements import center_of_mass

# Compatible with tensorflow backend
# https://arxiv.org/pdf/1708.02002.pdf


def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """

    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss

    return focal_loss


def sparse_categorical_accuracy(y_true, y_pred):
    """
    Accuracy metric for semantic image segmentation. None of the existing
    Keras accuracy metrics seem to work with the tensor shapes used here.
    Args:
        y_true: float32 array with true lables, shape: (-1, img_height * img_weidth)
        y_pred: float32 array with probabilities from a softmax layer, shape: (-1, img_height * img_weidth, nb_classes)
    Return:
        Accuracy of prediction
    """
    return K.cast(
        K.equal(y_true, K.cast(K.argmax(y_pred, axis=-1), K.floatx())), K.floatx()
    )


# https://gist.github.com/ilmonteux/8340df952722f3a1030a7d937e701b5a
def seg_metrics(
    y_true,
    y_pred,
    metric_name,
    metric_type="standard",
    drop_last=False,
    mean_per_class=False,
    verbose=False,
):
    """
    Compute mean metrics of two segmentation masks, via Keras.

    IoU(A,B) = |A & B| / (| A U B|)
    Dice(A,B) = 2*|A & B| / (|A| + |B|)

    Args:
        y_true: true masks, one-hot encoded.
        y_pred: predicted masks, either softmax outputs, or one-hot encoded.
        metric_name: metric to be computed, either 'iou' or 'dice'.
        metric_type: one of 'standard' (default), 'soft', 'naive'.
          In the standard version, y_pred is one-hot encoded and the mean
          is taken only over classes that are present (in y_true or y_pred).
          The 'soft' version of the metrics are computed without one-hot
          encoding y_pred.
          The 'naive' version return mean metrics where absent classes contribute
          to the class mean as 1.0 (instead of being dropped from the mean).
        drop_last = True: boolean flag to drop last class (usually reserved
          for background class in semantic segmentation)
        mean_per_class = False: return mean along batch axis for each class.
        verbose = False: print intermediate results such as intersection, union
          (as number of pixels).
    Returns:
        IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
          in which case it returns the per-class metric, averaged over the batch.

    Inputs are B*W*H*N tensors, with
        B = batch size,
        W = width,
        H = height,
        N = number of classes
    """

    flag_soft = metric_type == "soft"
    flag_naive_mean = metric_type == "naive"

    # always assume one or more classes
    num_classes = K.shape(y_true)[-1]

    if not flag_soft:
        # get one-hot encoded masks from y_pred (true masks should already be one-hot)
        y_pred = K.one_hot(K.argmax(y_pred), num_classes)
        y_true = K.one_hot(K.argmax(y_true), num_classes)

    # if already one-hot, could have skipped above command
    # keras uses float32 instead of float64, would give error down (but numpy arrays or keras.to_categorical gives float64)
    y_true = K.cast(y_true, "float32")
    y_pred = K.cast(y_pred, "float32")

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1, 2)  # W,H axes of each image
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    union = mask_sum - intersection  # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = 0.001
    iou = (intersection + smooth) / (union + smooth)
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)

    metric = {"iou": iou, "dice": dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask = K.cast(K.not_equal(union, 0), "float32")

    if drop_last:
        metric = metric[:, 1:]
        mask = mask[:, 1:]

    if verbose:
        print("intersection, union")
        print(K.eval(intersection), K.eval(union))
        print(K.eval(intersection / union))

    # return mean metrics: remaining axes are (batch, classes)
    if flag_naive_mean:
        return K.mean(metric)

    # take mean only over non-absent classes
    class_count = K.sum(mask, axis=0)
    non_zero = tf.greater(class_count, 0)
    non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
    non_zero_count = tf.boolean_mask(class_count, non_zero)

    if verbose:
        print("Counts of inputs with class present, metrics for non-absent classes")
        print(K.eval(class_count), K.eval(non_zero_sum / non_zero_count))

    return K.mean(non_zero_sum / non_zero_count)


def mean_iou(y_true, y_pred, **kwargs):
    """
    Compute mean Intersection over Union of two segmentation masks, via Keras.
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name="iou", **kwargs)


def mean_dice(y_true, y_pred, **kwargs):
    """
    Compute mean Dice coefficient of two segmentation masks, via Keras.
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """
    return seg_metrics(y_true, y_pred, metric_name="dice", **kwargs)


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def get_radio_old(
    pupil, max_r=True, min_t=0.0, max_t=1.0, kernel=np.ones((13, 13), dtype=np.uint8)
):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    # pupil 2d mask [0., 1.]
    pupil = pupil.astype(np.uint8)  # cv2 works with UMAT8
    # pupil = cv2.erode(pupil, kernel) # 3, 3 dio buen resultado
    pupil = cv2.morphologyEx(pupil, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(pupil, min_t, max_t)  # get edges
    # get contours
    cont, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # get contour with max area (assume pupil is the bigger object detected)
    cont = max(cont, key=cv2.contourArea)
    # redraw pupil
    pupil = np.zeros(pupil.shape, dtype=np.uint8)
    pupil = cv2.fillPoly(pupil, [cont], 1.0)
    # get center of biggest object detected
    center = center_of_mass(pupil)
    center = np.round(center).astype(np.int32)
    x, y = center
    # all x and y points
    xp, yp = np.where(pupil == 1)
    # largest distance between (xp, yp) and center is the radius
    rx, ry = np.abs((xp - x)).max(), np.abs((yp - y)).max()
    if max_r:
        r = max(rx, ry)
    else:
        r = min(rx, ry)
    return np.array([center[0], center[1], r])


def get_radio(mask, min_t=0.0, max_t=1.0, kernel=np.ones((13, 13), dtype=np.uint8)):
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

    # mask 2d array with [0., 1.]
    mask = mask.astype(np.uint8)  # cv2 works with UMAT8
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edged = cv2.Canny(mask, min_t, max_t)  # get edges
    # get contours
    cont, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # get contour with max area (assume the bigger object is the final object)
    cont = max(cont, key=cv2.contourArea)
    # redraw mask
    mask = np.zeros(mask.shape, dtype=np.uint8)
    mask = cv2.fillPoly(mask, [cont], 1.0)
    # get center of biggest object detected
    center = center_of_mass(mask)
    center = np.round(center).astype(np.int32)
    x, y = center
    # all x and y points
    xp, yp = np.where(mask == 1)
    # largest distance between (xp, yp) and center is the radius
    rx, ry = np.abs((xp - x)).max(), np.abs((yp - y)).max()

    return np.array([center[0], center[1], min(rx, ry), max(rx, ry)])


def distance_error(label, mask):
    lb_coords = get_radio(label)
    pred_coords = get_radio(mask)

    xy_lb, xy_pd = lb_coords[:2], pred_coords[:2]
    r_lb, r_pd = lb_coords[-1], pred_coords[-1]

    euc_dist = np.linalg.norm(xy_lb - xy_pd)
    radio_diff = np.abs(r_lb - r_pd)
    # radio_diff = np.abs( 1 - np.divide(r_pd, r_lb, where = r_lb != 0) )

    return euc_dist, radio_diff
