import cv2
import time
import logging
import numpy as np
import configparser
from tqdm import tqdm
from os.path import abspath
from scipy.io import loadmat
from metrics import mean_iou, mean_dice
from tensorflow.keras.models import load_model
from tensorflow.python.keras.models import Model
from scipy.ndimage.measurements import center_of_mass
from tensorflow.python.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPool2D,
    Flatten,
)

from abstract_segmentator import AbstractSegmentatorClass

# TODO: add more documentation

logger = logging.getLogger("dense10_20220629")


class DenseSegmentator(AbstractSegmentatorClass):
    """Eye Segmentator
    0 --> Background
    1 --> Iris
    2 --> Pupil
    3 --> Sclera
    """

    def __init__(self, *args, modelpath="", inifile_path="", **kwargs):
        """Class instantiation.

        Parameters
        ----------
        modelpath : str
            Path to hdf5 model.
        """
        config = configparser.ConfigParser()
        config.read(inifile_path)

        # segmentator config
        self.iris_id = config.getint("Segmentator", "IrisID")
        self.pupil_id = config.getint("Segmentator", "PupilID")
        # self.sclera_id = config.getint("Segmentator", "ScleraID") # not used
        self.periocular_id = config.getint("Segmentator", "PeriocularID")
        self.rtype = config.get("Segmentator", "RadiusType").lower()
        threshold = config.getfloat("Segmentator", "Threshold")

        # iris modularization config
        self.max_shift = config.getint("IrisEncode", "MaxShift")
        self.iris_encode_type = config.get("IrisEncode", "EncodeType").lower()
        self.iris_mod_type = config.get("IrisEncode", "modType").lower()
        self.icapath = config.get("IrisEncode", "ICAPath")
        self.rubbersheet_height = config.getint("IrisEncode", "RubbersheetHeight")
        self.rubbersheet_width = config.getint("IrisEncode", "RubbersheetWidth")

        # additional data for iris mod
        self.radii = np.arange(0, 1, 1 / self.rubbersheet_height)
        self.angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.rubbersheet_width)
        self.cos_angles = np.cos(self.angles)
        self.sin_angles = np.sin(self.angles)

        kwargs.update(
            {
                "modelpath": modelpath,
                "threshold": threshold,
            }
        )

        super().__init__(*args, **kwargs)

        # radii estimators config
        self.radii_estimators = self.set_radii_estimators()

        # additional data config
        self.icamat = None  # TODO: parametrize this variable

        # logger config
        logger.debug("DenseSegmentator model loaded")

    def set_model(self, *args, **kwargs):
        model = load_model(
            self.modelpath,
            compile=False,
            custom_objects={
                "mean_iou": mean_iou,
                "mean_dice": mean_dice,
            },
        )

        # make segmentator
        model = Model(
            name="Dense10_Segmentator",
            inputs=model.input,
            outputs=[
                model.output,
            ],
        )

        # load iris encode type
        if self.iris_encode_type == "ica":
            self.icamat = loadmat(abspath(self.icapath))["ICAtextureFilters"]
        else:
            logger.warning("No iris encode type selected")
            self.icamat = None

        return model

    def get_target_size(self, *args, model=None, **kwargs):
        target_size = self.model.output_shape[1:-1][::-1]
        return target_size

    def summary(self):
        return self.model.summary()

    def count_params(self):
        return self.model.count_params()

    def _center_of_mass(
        self,
        *args,
        mask=None,
        min_t=0.0,
        max_t=1.0,
        kernel=np.ones((13, 13), dtype=np.uint8),
        return_circle=True,
        **kwargs,
    ):
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        try:
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

            if return_circle:  # this will return a circle with both min radii detected
                return np.array([center[0], center[1], min(rx, ry), min(rx, ry)])
            else:  # this will return an ellipse with min and max radii detected
                return np.array([center[0], center[1], min(rx, ry), max(rx, ry)])

        except ValueError as er:
            # log error
            return np.zeros(4)

    def imfill_holes(self, bw):
        # Copy the thresholded image.
        im_floodfill = bw.copy()
        # Mask used to flood filling
        h, w = bw.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        return bw | im_floodfill_inv

    def circfit(self, x, y):
        # By: Izhak bucher 25/oct /1991,
        x = np.reshape(x.flatten(), (-1, 1))
        y = np.reshape(y.flatten(), (-1, 1))
        A = np.concatenate((x, y, np.ones(x.shape)), 1)
        B = -(x ** 2) - y ** 2
        a, _, _, _ = np.linalg.lstsq(A, B)
        xc = float(-0.5 * a[0])
        yc = float(-0.5 * a[1])
        R = float(np.sqrt(0.25 * (a[0] ** 2 + a[1] ** 2) - a[2]))
        return xc, yc, R

    def LMS2(
        self,
        *args,
        mask=None,
        dy=np.array(
            [
                [-1],
                [0],
                [1],
            ]
        ),
        kernel=np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        ),
        **kwargs,
    ):
        def im_contour(bw, kernel):
            eroded = cv2.erode(bw, kernel, iterations=1)
            return bw - eroded

        # Isolate iris and pupil
        iris = self.imfill_holes(mask)
        pupil = cv2.bitwise_xor(mask, iris)

        # Test if pupil was found
        y, x = np.where(pupil > 0)

        if len(x) > 5:  # If pupil was found
            # Obtain pupil's center of mass
            xp = np.mean(x)
            yp = np.mean(y)

            # Pupil radii along x and y axes
            rx = (np.max(x) - np.min(x)) / 2 + 1
            ry = (np.max(y) - np.min(y)) / 2 + 1

            # Pupil coordinates
            # xp, yp, rx = circfit(x,y)
            pupil_xyr = np.array([xp, yp, rx, ry])  # daniel
            # pupil_xyr = np.array([yp, xp, min(rx, ry), max(rx, ry)]) # own
            # pupil_xyr = np.array([yp, xp, ry, rx]) # own

            # Obtain iris contour
            iris = im_contour(iris, kernel)
            y, x = np.where(iris > 0)

            # Find radious along the y axis
            ry = (np.max(y) - np.min(y)) / 2

            # Remove eyeleads from iris contour
            max_x = np.max(x)
            min_x = np.min(x)
            box = int(0.07 * (max_x - min_x))
            iris[:, min_x + box : max_x - box] = 0

            # Obtain the circle that best adapts the valid iris contour
            y, x = np.where(iris > 0)
            xi, yi, rx = self.circfit(x, y)

            # Iris coordinates
            iris_xyr = np.array([xi, yi, rx, ry])  # daniel
            # iris_xyr = np.array([yi, xi, min(rx, ry), max(rx, ry)]) # own
            # iris_xyr = np.array([yi, xi, ry, rx]) # own

        else:  # pupil was not found

            # Find horizontal lines (eyeleads)
            h_lines = cv2.filter2D(mask, -1, dy)

            # Find valid iris contour
            iris = im_contour(mask, kernel)
            iris = iris - h_lines
            y, x = np.where(iris > 0)

            # Test if there is an iris
            if len(x) > 50:
                # Find an initial aproximation of the center
                cx = np.mean(x)
                cy = np.mean(y)

                # Find initial approximations of iris size
                min_y = np.min(y)
                ry = (np.max(y) - min_y) / 2
                max_x = np.max(x)
                min_x = np.min(x)

                # Estimate aproximate pupil diameter
                dp = int(0.5 * (max_x - min_x))

                # Separate pupil from iris
                contour = iris.copy()
                iris[min_y : min_y + dp, int(cx - dp / 2) : int(cx + dp / 2)] = 0
                pupil = cv2.bitwise_xor(contour, iris)

                # Clean iris a bit more
                iris[:, int(cx - 0.6 * dp) : int(cx + 0.6 * dp)] = 0

                # Obtain the circle that best adapts the valid pupil contour
                y, x = np.where(pupil > 0)
                xp, yp, rp = self.circfit(self, x, y)

                # Pupil coordinates
                pupil_xyr = np.array([xp, yp, rp, rp])  # daniel
                # pupil_xyr = np.array([yp, xp, rp, rp]) # own

                # Obtain the circle that best adapts the valid iris contour
                y, x = np.where(iris > 0)
                xi, yi, rx = self.circfit(self, x, y)

                # Iris coordinates
                iris_xyr = np.array([xi, yi, rx, ry])  # daniel
                # iris_xyr = np.array([yi, xi, min(rx, ry), max(rx, ry)]) # own
                # iris_xyr = np.array([yi, xi, ry, rx]) # own

            else:
                pupil_xyr = np.zeros(4)
                iris_xyr = np.zeros(4)

        return pupil_xyr, iris_xyr  # daniel

    def LMS3(self, mask=None, *args, **kwargs):

        # add internal functions of lms3
        def kernel7():
            kernel = np.ones([7, 7], dtype=np.uint8)

            for i in (0, 1, 5, 6):
                for j in (0, 1, 5, 6):
                    kernel[j, i] = 0

            for i in (1, 5):
                for j in (1, 5):
                    kernel[j, i] = 1

            return kernel

        def get_biggest(mask):
            # Find area of connected components
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            sizes = stats[1:, -1]
            if len(sizes) > 0:
                id_max = np.argmax(sizes) + 1
            else:
                id_max = 0

            # Choose the region with the biggest area
            if len(sizes) > 0:
                mask = output == id_max

            return mask.astype("uint8") * 255

        def find_horizontal_lines(mask, kernel):
            dy = np.array([[-1], [0], [1]])
            h_lines = np.abs(cv2.filter2D(mask, cv2.CV_16S, dy)).astype(np.uint8)
            h_lines = cv2.GaussianBlur(h_lines, (5, 5), 2) > 100
            h_lines = cv2.dilate(h_lines.astype(np.uint8), kernel)
            return h_lines

        def im_contour(bw, kernel):
            eroded = cv2.erode(bw, kernel, iterations=1)
            return bw - eroded

        # define kernels to use
        se1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        se2 = kernel7()

        # follow daniel flux
        iris = mask[..., self.iris_id]
        pupil = mask[..., self.pupil_id]

        iris = iris.astype(np.uint8)
        pupil = pupil.astype(np.uint8)

        # Get biggest area
        iris = get_biggest(iris)
        pupil = get_biggest(pupil)

        # Clean iris
        iris = self.imfill_holes(iris)
        iris_mask = iris
        iris_mask[pupil > 0] = 0

        # Morphological Open
        iris = cv2.morphologyEx(iris, cv2.MORPH_OPEN, se1)
        pupil = cv2.morphologyEx(pupil, cv2.MORPH_OPEN, se1)

        # Morphological Close
        iris = cv2.morphologyEx(iris, cv2.MORPH_CLOSE, se1)
        pupil = cv2.morphologyEx(pupil, cv2.MORPH_CLOSE, se1)

        # Iris radius on the y axis
        y, _ = np.where(iris)
        ri_y = (max(y) - min(y)) / 2

        # Pupil radius on the y axis
        y, _ = np.where(pupil)
        try:
            rp_y = (max(y) - min(y)) / 2
        except ValueError as ve:
            logger.error("An error ocurred getting minmax(y) inside LMS3")
            rp_y = 0

        # Find eye leads
        parp1 = find_horizontal_lines(pupil, se1)
        parp2 = find_horizontal_lines(iris, se1)

        # Find Pupil contour
        pupil2 = im_contour(pupil, se1)
        pupil2[parp1 > 0] = 0

        # Find Iris contour
        iris2 = im_contour(iris, se1)
        iris2[parp2 > 0] = 0
        iris2[cv2.dilate(pupil, se2) > 0] = 0

        # Find pupil circle
        y, x = np.where(pupil2 > 0)
        xp, yp, rp_x = self.circfit(x, y)

        # Find iris circle
        y, x = np.where(iris2 > 0)
        xi, yi, ri_x = self.circfit(x, y)

        # Outputs
        pupil_xyr = np.array([xp, yp, rp_x, rp_y]).astype(np.int32)
        iris_xyr = np.array([xi, yi, ri_x, ri_y]).astype(np.int32)

        # return pupil_xyr, iris_xyr, iris_mask # daniel
        return pupil_xyr, iris_xyr

    # how to add more estimators?
    # example below
    def set_radii_estimators(self, *args, **kwargs):
        return {
            "center_of_mass": self._center_of_mass,
            "lms2": self.LMS2,
            "lms3": self.LMS3,
            # "another_key": self.another_estimator
        }

    def get_radio(self, mask, rtype="center_of_mass", *args, **kwargs):
        if rtype not in list(self.radii_estimators.keys()):
            logger.error(f"{rtype} function not defined.")
            return np.zeros(4)
        else:
            _radii_function = self.radii_estimators[rtype.lower()]
            return _radii_function(mask=mask, *args, **kwargs)

    def distance_error(self, info, label, dist_type="", *args, **kwargs):
        if dist_type.lower() == "pupil":
            pupil_label = label[..., self.pupil_id]
            pupil_lb_coords = self.get_radio(*args, mask=pupil_label, **kwargs)

            pupil_pred_coords = [
                info["pupil_x"],
                info["pupil_y"],
                info["pupil_r_min"],
                info["pupil_r_max"],
            ]

            xy_lb, xy_pd = pupil_lb_coords[:2], pupil_pred_coords[:2]
            min_r_lb, min_r_pd = pupil_lb_coords[-2], pupil_pred_coords[-2]
            max_r_lb, max_r_pd = pupil_lb_coords[-1], pupil_pred_coords[-1]

            center_euc_dist = np.linalg.norm(xy_lb - xy_pd)
            min_radio_diff = np.abs(min_r_lb - min_r_pd)
            max_radio_diff = np.abs(max_r_lb - max_r_pd)

            return {
                "center_diff": center_euc_dist,
                "min_radio_diff": min_radio_diff,
                "max_radio_diff": max_radio_diff,
            }

        elif dist_type.lower() == "iris":
            # get (iris | pupil) predicted mask
            iris_mask = info["mask"][..., self.iris_id].astype(np.bool)
            pupil_mask = info["mask"][..., self.pupil_id].astype(np.bool)
            iris_mask = np.logical_or(iris_mask, pupil_mask)
            iris_mask = iris_mask.astype(np.uint8)

            # get (iris | pupil) label mask
            iris_label = label[..., self.iris_id].astype(np.bool)
            pupil_label = label[..., self.pupil_id].astype(np.bool)
            iris_label = np.logical_or(iris_label, pupil_label)
            iris_label = iris_label.astype(np.uint8)

            # get radio of pred mask and label
            iris_pred_coords = self.get_radio(*args, mask=iris_mask, **kwargs)

            iris_lb_coords = self.get_radio(*args, mask=iris_label, **kwargs)

            # calculate distances
            xy_lb, xy_pd = iris_lb_coords[:2], iris_pred_coords[:2]
            min_r_lb, min_r_pd = iris_lb_coords[-2], iris_pred_coords[-2]
            max_r_lb, max_r_pd = iris_lb_coords[-1], iris_pred_coords[-1]

            center_euc_dist = np.linalg.norm(xy_lb - xy_pd)
            min_radio_diff = np.abs(min_r_lb - min_r_pd)
            max_radio_diff = np.abs(max_r_lb - max_r_pd)

            return {
                "center_diff": center_euc_dist,
                "min_radio_diff": min_radio_diff,
                "max_radio_diff": max_radio_diff,
            }

    # iris modularization function for iris recognition
    def get_rubbersheet(self, image, pupil_xyr, iris_xyr):
        # Angle value
        rs = np.zeros((self.rubbersheet_height, self.rubbersheet_width, 3), np.uint8)

        if self.iris_mod_type == "height":
            # decode centers and radii
            cx, cy, pupil_r, _ = pupil_xyr
            _, _, iris_r, _ = iris_xyr

            for i in range(self.rubbersheet_height):
                rad = i / self.rubbersheet_height

                x_lowers = cx + pupil_r * self.cos_angles
                y_lowers = cy + pupil_r * self.sin_angles
                x_uppers = cx + iris_r * self.cos_angles
                y_uppers = cy + iris_r * self.sin_angles

                # Fill in the rubbersheet
                Xc = (1 - rad) * x_lowers + rad * x_uppers
                Yc = (1 - rad) * y_lowers + rad * y_uppers

                rs[i, ...] = image[Xc.astype(int), Yc.astype(int)]

        else:
            # decode centers and radii
            yp, xp, rp, _ = pupil_xyr
            yi, xi, ri, _ = iris_xyr

            for i in range(self.rubbersheet_width):
                rad = i / self.rubbersheet_width

                x_lowers = xp + rp * self.cos_angles[i]
                y_lowers = yp + rp * self.sin_angles[i]
                x_uppers = xi + ri * self.cos_angles[i]
                y_uppers = yi + ri * self.sin_angles[i]

                # fill in the rubbersheet
                Xc = (1 - self.radii) * x_lowers + self.radii * x_uppers
                Yc = (1 - self.radii) * y_lowers + self.radii * y_uppers

                rs[:, i, :] = image[Yc.astype(int), Xc.astype(int)]

        return rs

    def predict(self, image, original_shape=(None, None)):
        """Infer the mask of the input image. The image to be
        masked must first be processed using the process_image()
        function. The image is reshaped and normalized before inference.

        Parameters
        ----------
        image : numpy.ndarray
            Image to segment.
        original_shape : tuple
            Original image size, unknown by default (None, None)

        Returns
        -------
        numpy.ndarray
            Image mask with size equal to original size.
        """
        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=-1)

        image = np.expand_dims(image, axis=0)
        pred_mask = self.model.predict(image)
        pred_mask = pred_mask[0, ...]  # obtain unique element from batch
        pred_shape = pred_mask.shape

        if original_shape != (None, None):
            pred_mask = cv2.resize(pred_mask, original_shape[::-1])

        # process mask
        if self.threshold == 0.0:
            argmax_mask = np.argmax(pred_mask, axis=-1)
            for c in range(pred_mask.shape[-1]):
                pred_mask[..., c] = np.where(argmax_mask == c, 1.0, 0.0)
            logger.debug(f"Mask thresholded with np.argmax function")
        else:
            for c in range(pred_mask.shape[-1]):
                pred_mask[..., c] = pred_mask[..., c] >= self.threshold
            logger.debug(f"Mask thresholded with {self.threshold}")

        logger.debug(f"Predicted mask shape: {pred_shape}")
        logger.debug(f"Resized mask shape: {pred_mask.shape}")

        pred_mask = pred_mask.astype(np.uint8)

        return pred_mask

    def forward(self, image=None, verbose=False, *args, **kwargs):
        """Apply sequentially all model methods to get a final
        bitwised and cropped image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image.

        Returns
        -------
        Dict with keys:
            - "pupil_x": x coordinate of the pupil center
            - "pupil_y": y coordinate of the pupil center
            - "pupil_r_min": min radius of the pupil
            - "pupil_r_max": max radius of the pupil
            - "iris_x": x coordinate of the iris center
            - "iris_y": y coordinate of the iris center
            - "iris_r_min": min radius of the iris
            - "iris_r_max": max radius of the iris
            - "mask": all masks with the same size of the input image. Resized mask where each channel correspond to each eye segment of input image.
            - "iris_rubbersheet": modularized iris as rgb image with same type as input image
            - "pred_shape": shape of the predicted mask.
            - "radii_type_estimator": radiuses type estimator used.
            - "original_shape": original shape of the input image.
        """

        r_pupil = [
            0,
        ] * 4
        r_iris = [
            0,
        ] * 4

        image_processed, original_shape = self.process_image(
            image, tsize=self.target_size
        )

        mask = self.predict(image_processed, original_shape)

        pupil_mask = mask[..., self.pupil_id]
        iris_mask = mask[..., self.iris_id]

        # avg pupil and iris center of masks
        if self.rtype.lower() == "center_of_mass":
            try:  # obtain pupil radio
                # may raise error if no pupil was detected
                r_pupil = self.get_radio(
                    mask=pupil_mask,
                    rtype=self.rtype,
                    kernel=np.ones((3, 3), dtype=np.uint8),
                )
                if verbose:
                    logger.warn(f"pupil info: {r_pupil}")
            except Exception as ex:
                if verbose:
                    logger.warn("No pupil detected")
                    logger.warn(ex)

            try:  # obtain iris radio
                # may raise error if no iris was detected
                iris_mask = np.logical_or(
                    iris_mask.astype(np.bool), pupil_mask.astype(np.bool)
                ).astype(np.uint8)
                r_iris = self.get_radio(
                    mask=iris_mask,
                    rtype=self.rtype,
                    kernel=np.ones((7, 7), dtype=np.uint8),
                )
                if verbose:
                    logger.warn(f"iris info: {r_iris}")
            except Exception as ex:
                if verbose:
                    logger.warn("No iris detected")
                    logger.warn(ex)

        # mean squared error v2_DB
        elif self.rtype.lower() == "lms2":
            r_pupil, _ = self.get_radio(
                mask=pupil_mask,
                rtype=self.rtype,
                kernel=np.ones((3, 3), dtype=np.uint8),
            )
            _, r_iris = self.get_radio(
                mask=iris_mask, rtype=self.rtype, kernel=np.ones((3, 3), dtype=np.uint8)
            )
            if verbose:
                logger.warn(f"pupil info: {r_pupil}")
                logger.warn(f"iris info: {r_iris}")

            # fix daniel radiis
            r_pupil[0], r_pupil[1] = r_pupil[1], r_pupil[0]
            r_iris[0], r_iris[1] = r_iris[1], r_iris[0]

        # hardcoded because this have a different behaviour
        # mean squared error v3_DB
        elif self.rtype.lower() == "lms3":
            try:
                pupil_xyr, iris_xyr = self.LMS3(mask)
                # fix daniel radii
                r_pupil[0], r_pupil[1] = pupil_xyr[1], pupil_xyr[0]
                r_iris[0], r_iris[1] = iris_xyr[1], iris_xyr[0]
                # assign radii to remaning slots in r_pupil and r_iris
                # in lms3, DB said the function will return a circle, not an ellipse
                r_pupil[2] = pupil_xyr[2]
                r_pupil[3] = pupil_xyr[3]

                r_iris[2] = iris_xyr[2]
                r_iris[3] = iris_xyr[3]

                if verbose:
                    logger.warn(f"pupil info: {r_pupil}")
                    logger.warn(f"iris info: {r_iris}")
            except ValueError as ve:
                pass

        # after getting radii of pupil and iris, we can
        # modularize iris to get iris code from it
        # TODO: call functions here
        iris_rubsheet = self.get_rubbersheet(image, r_pupil, r_iris)

        elem = {
            "pupil_x": int(r_pupil[0]),
            "pupil_y": int(r_pupil[1]),
            "pupil_r_min": int(r_pupil[2]),
            "pupil_r_max": int(r_pupil[3]),
            "iris_x": int(r_iris[0]),
            "iris_y": int(r_iris[1]),
            "iris_r_min": int(r_iris[2]),
            "iris_r_max": int(r_iris[3]),
            "mask": mask,
            "iris_rubbersheet": iris_rubsheet,
            "pred_shape": list(self.target_size),
            "radii_type_estimator": self.rtype.lower(),
            "original_shape": list(original_shape),
        }

        return elem

    # eval functions
    # -------------------------------------------------------------------------

    def eval_generator_radii(
        self,
        *args,
        generator=None,
        verbose=False,
        desc="",
        rtype="center_of_mass",
        return_bg=False,
        **kwargs,
    ):
        results = []

        if verbose:
            _iter = tqdm(range(len(generator)), desc=desc)

        for i in _iter:
            fname = generator.image_info[i]["path"]
            image, label = generator[i]
            image = (image[0] * 255).astype(np.uint8)
            start = time.time()
            info = self.forward(image=image, rtype=rtype)
            end = time.time()
            label = label[0, ...]

            iou = self.eval_iou(
                info["mask"], label, generator.class_info, return_bg=return_bg
            )

            iris_dist_err = self.distance_error(
                info=info,
                label=label,
                dist_type="iris",
            )

            pupil_dist_err = self.distance_error(
                info=info,
                label=label,
                dist_type="pupil",
            )

            info.pop("mask")

            info.update(
                {
                    "fname": fname,
                    "miou": iou["miou"],
                    "inf_time": end - start,
                    "_classes": {
                        c["name"]: iou[c["name"]] for c in generator.class_info
                    },
                }
            )

            info.update({"iris_" + key: iris_dist_err[key] for key in iris_dist_err})

            info.update({"pupil_" + key: pupil_dist_err[key] for key in pupil_dist_err})

            results.append(info)

        return results

    # eval current model with a validation dataset and return results in a dictionary
    def eval_generator(self, generator=None, verbose=False, desc="", return_bg=False):
        results = []

        if verbose:
            _iter = tqdm(range(len(generator)), desc=desc)

        for i in _iter:
            fname = generator.image_info[i]["path"]
            image, label = generator[i]
            start = time.time()
            pred_mask, _ = self.model.predict(image)
            end = time.time()
            pred_mask = pred_mask[0, ...]  # obtain unique element from batch
            label = label[0, ...]
            iou = self.eval_iou(
                pred_mask, label, generator.class_info, return_bg=return_bg
            )

            results.append(
                {
                    "fname": fname,
                    "miou": iou["miou"],
                    "inf_time": end - start,
                    "_classes": {
                        c["name"]: iou[c["name"]] for c in generator.class_info
                    },
                }
            )

        return results
