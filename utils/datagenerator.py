import os
import cv2
import json
import glob
import imgaug
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from .utils import load_image
from imutils.paths import list_images
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class EyesDataset(Dataset):
            def load_eyes(self):
                    ...
            def load_mask(self, image_id):
                    ...
            def image_reference(self, image_id):
                    ...
    """

    def __init__(
        self,
        shuffle=True,
        dim=(480, 640),
        color_space="RGB",
        augmentation=None,
        channels=3,
        batch_size=1,
        class_map=None,
        preprocess=[],
    ):
        self.dim = dim
        self._image_ids = []
        self.image_info = []
        self.shuffle = shuffle
        self.channels = channels
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.input_shape = (*dim, channels)
        self.class_info = []
        self.preprocess = preprocess
        self.source_class_ids = {}
        self.color_space = color_space

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
                # "color": color,
            }
        )

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use."""

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        self.class_map = class_map

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i["source"] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info["source"] == source
        return info["id"]

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_ids):
        """Load the specified image and return a [BS,H,W,3] Numpy array."""
        bs = np.zeros([self.batch_size, *self.dim, self.channels], np.uint8)

        for i, idx in enumerate(image_ids):
            # Load image
            image = load_image(
                self.image_info[idx]["path"], colorspace=self.color_space
            )
            image = cv2.resize(image, self.dim[::-1])  # fuck you cv2

            # asign image to batch
            bs[
                i,
            ] = image

        return bs

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
                masks: A bool array of shape [height, width, instance count] with
                        a binary mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning(
            "You are using the default load_mask(), maybe you need to define your own one."
        )
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle == True:
            np.random.shuffle(self._image_ids)

    def __len__(self):
        raise NotImplementedError("abstract method '__len__' not implemented")

    def __getitem__(self, index):
        raise NotImplementedError("abstract method '__getitem__' not implemented")


class EyeDataset(Dataset):
    def load_eyes(self, dataset_dir, subset):
        """Load a subset of the Eye dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("eye", 0, "eye")
        self.add_class("eye", 1, "iris")
        self.add_class("eye", 2, "pupil")
        self.add_class("eye", 3, "sclera")
        # self.add_class("eye", 4, "eye")

        # Train, test or validation dataset?
        assert subset in ["train", "test", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        """
		# Load annotations
		# regions = list of regions
		# regions:
		# {
		# 	image_name: [
		# 		{
		# 			'shape_attributes': {
		# 				'name': 'polygon', 
		# 				'all_points_x': [...], 
		# 				'all_points_y': [...]
		# 			}, 
		# 			'region_attributes': {'Eye': 'iris'}
		# 		}
		# 		...
		# 	]
		# }
		"""

        annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))

        # Add images
        for key in annotations:
            # key = image_name
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            polygons = [r["shape_attributes"] for r in annotations[key]]
            objects = [s["region_attributes"] for s in annotations[key]]

            # num_ids = [1, 2, 3] => ['iris', 'pupil', 'sclera', ]
            num_ids = []

            for obj in objects:
                for cl_info in self.class_info:
                    if cl_info["name"] == obj["Eye"]:
                        num_ids.append(cl_info["id"])

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, key)
            # height, width = skimage.io.imread(image_path).shape[:2]
            # Pillow use less memory
            width, height = Image.open(image_path).size

            self.add_image(
                "eye",
                image_id=key,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids,
            )

    def load_mask(self, image_ids):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [bs, height, width, instance count] with
                one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        bs_mask = np.zeros(
            [self.batch_size, *self.dim, self.num_classes], dtype=np.float32
        )

        for idx, imid in enumerate(image_ids):
            # If not an eye dataset image, delegate to parent class.
            image_info = self.image_info[imid]
            if image_info["source"] != "eye":
                return super(self.__class__, self).load_mask(imid)
            num_ids = image_info["num_ids"]

            # Convert polygons to a bitmap mask of shape
            # [height, width, instance_count]
            info = self.image_info[imid]
            mask = np.zeros(
                [image_info["height"], image_info["width"], self.num_classes],
                dtype=np.bool,
            )

            for i, p in zip(num_ids, image_info["polygons"]):
                # Get indexes of pixels inside the polygon and set them to 1
                try:
                    if p["name"] in [
                        "polygon",
                        "polyline",
                    ]:
                        rr, cc = skimage.draw.polygon(
                            p["all_points_y"], p["all_points_x"]
                        )
                    elif p["name"] in [
                        "ellipse",
                    ]:
                        rr, cc = skimage.draw.ellipse(
                            p["cy"], p["cx"], p["ry"], p["rx"]
                        )
                    elif p["name"] in [
                        "circle",
                    ]:
                        rr, cc = skimage.draw.circle(p["cy"], p["cx"], p["r"])

                    mask[rr, cc, i] = True

                except (KeyError, IndexError) as ex:
                    print(image_info, i, p, ex)

            # fix to iris with pupil
            id_pupil = next(d["id"] for d in self.class_info if "pupil" in d["name"])
            id_iris = next(d["id"] for d in self.class_info if "iris" in d["name"])
            id_eye = next(d["id"] for d in self.class_info if "eye" in d["name"])
            pupil = mask[..., id_pupil].copy().astype(np.bool)
            iris = mask[..., id_iris].copy().astype(np.bool)
            iris_xor = np.logical_xor(iris, pupil)
            iris_xor = np.where(iris_xor == True, 1.0, 0.0)
            mask[..., id_iris] = iris_xor.copy()

            one_mask = np.argmax(mask, axis=-1)
            mask[..., id_eye] = np.where(one_mask == id_eye, 1.0, 0.0)
            mask = cv2.resize(mask.astype(np.uint8), self.dim[::-1])
            mask = mask.astype(np.bool)

            bs_mask[
                idx,
            ] = mask.astype(np.float32)

        return bs_mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "eye":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.num_images // self.batch_size

    def dataAugmentation(self, image, masks):
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = [
            "KeepSizeByResize",
            "CropToFixedSize",
            "TranslateX",
            "TranslateY",
            "Pad",
            "Lambda",
            "Sequential",
            "SomeOf",
            "OneOf",
            "Sometimes",
            "Affine",
            "PiecewiseAffine",
            "CoarseDropout",
            "Fliplr",
            "Flipud",
            "CropAndPad",
            "PerspectiveTransform",
        ]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        for bs in range(self.batch_size):
            # Store shapes before augmentation to compare
            image_shape = image[
                bs,
            ].shape
            mask_shape = masks[
                bs,
            ].shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = self.augmentation.to_deterministic()
            image[bs,] = det.augment_image(
                image[
                    bs,
                ]
            )

            # for each mask, slow?
            # for c in range(masks[bs, ].shape[-1]):
            # 	uint8_mask = masks[bs, ..., c].astype(np.uint8)

            # 	masks[bs, ..., c] = det.augment_image(
            # 		uint8_mask, hooks=imgaug.HooksImages(activator=hook)
            # 	).astype(np.float32)

            # in one shot
            masks[bs, ...] = det.augment_image(
                masks[bs, ...].astype(np.uint8),
                hooks=imgaug.HooksImages(activator=hook),
            ).astype(np.float32)

            # Verify that shapes didn't change
            assert (
                image[
                    bs,
                ].shape
                == image_shape
            ), "Augmentation shouldn't change image size"
            assert (
                masks[
                    bs,
                ].shape
                == mask_shape
            ), "Augmentation shouldn't change mask size"

        return image, masks

    def __getitem__(self, index):
        if index > self._image_ids.max():
            raise IndexError(
                f"List index out of range. Size of generator: {self.__len__()}"
            )
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self._image_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images = self.load_image(indexes)
        masks = self.load_mask(indexes)

        if self.preprocess:
            for func in self.preprocess:
                for bs in range(self.batch_size):
                    images[bs, ...] = func(images[bs, ...])

        # data augmentation
        if self.augmentation:
            images, masks = self.dataAugmentation(images, masks)

        images = images / 255.0

        return images, masks

    def get_image_by_name(self, imname):
        assert self.batch_size == 1, "Batch size must be 1."
        for i in range(len(self._image_ids)):
            if imname in self.image_reference(i):
                return self.__getitem__(i)


class OpenEDS(Dataset):
    def add_class(self, source, class_id, class_name, color):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
                "color": color,
            }
        )

    def load_eyes(self, dataset_dir, subset):
        """Load a subset of the Eye dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # read and parse csv file
        df = pd.read_csv(os.path.join(dataset_dir, "class_dict.csv"))

        # Add classes
        self.add_class("eye", 0, "bg", df.T[0].values[1:])
        self.add_class("eye", 1, "sclera", df.T[1].values[1:])
        self.add_class("eye", 2, "iris", df.T[2].values[1:])
        self.add_class("eye", 3, "pupil", df.T[3].values[1:])

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        # dataset_dir = os.path.join(dataset_dir, subset)

        # images = [
        #     os.path.join(dataset_dir, "images", x)
        #     for x in sorted(os.listdir(os.path.join(dataset_dir, "images")))
        # ]
        # labels = [
        #     os.path.join(dataset_dir, "labels", x)
        #     for x in sorted(os.listdir(os.path.join(dataset_dir, "labels")))
        # ]

        images = sorted([*list_images(os.path.join(dataset_dir, subset, "images"))])
        labels = sorted([*list_images(os.path.join(dataset_dir, subset, "labels"))])

        # split images and labels into train, test and val

        # Add images
        for impath, lbpath in tqdm(
            zip(images, labels), desc=f"Loading {subset} images"
        ):
            imname = os.path.basename(impath)
            # lb = np.load(lbpath)
            lb = load_image(lbpath)
            # image = load_image(impath)
            # height, width = image.shape[:2]
            image = Image.open(impath)
            width, height = image.size

            self.add_image(
                "eye",
                image_id=imname,
                path=impath,
                width=width,
                height=height,
                lbpath=lbpath,
            )

    def load_mask(self, image_ids):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        bs_mask = np.zeros(
            [self.batch_size, *self.dim, self.num_classes], dtype=np.float32
        )

        for idx, imid in enumerate(image_ids):
            # If not an eye dataset image, delegate to parent class.
            image_info = self.image_info[imid]
            if image_info["source"] != "eye":
                return super(self.__class__, self).load_mask(imid)

            # Convert polygons to a bitmap mask of shape
            # [height, width, instance_count]
            # lb = np.load(image_info["lbpath"])
            try:
                lb = load_image(image_info["lbpath"])

                # each object in self.class_info have his own color, so we can parse the label image
                # sintax:
                # [{'source': 'eye', 'id': 0, 'name': 'bg'},
                # {'source': 'eye', 'id': 1, 'name': 'sclera'},
                # {'source': 'eye', 'id': 2, 'name': 'iris'},
                # {'source': 'eye', 'id': 3, 'name': 'pupil'}]

                for obj in self.class_info:
                    class_idx = obj["id"]
                    # color = obj["color"] # rgb
                    # create mask
                    if obj["name"] == "pupil":
                        wh = np.where(lb[..., 1] == 255)
                    elif obj["name"] == "iris":
                        wh = np.where(lb[..., 1] == 103)
                    elif obj["name"] == "sclera":
                        wh = np.where(lb[..., 2] == 255)
                    else:  # bg
                        wh = np.where(lb[..., 1] == 0)

                    mask = np.zeros(lb.shape[:-1], dtype=np.uint8)
                    mask[wh] = 1
                    # then resize
                    mask = cv2.resize(mask, self.dim[::-1])
                    bs_mask[idx, ..., class_idx] = mask.astype(np.float32)

            except Exception as ex:
                print(f'Error in image {image_info["id"]}')
                print(ex)
                bs_mask[idx, ...] = np.zeros(
                    [*self.dim, self.num_classes],
                    dtype=np.float32,
                )

        return bs_mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "eye":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def dataAugmentation(self, image, masks):
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = [
            "Sequential",
            "SomeOf",
            "OneOf",
            "Sometimes",
            "Fliplr",
            "Flipud",
            "CropAndPad",
            "Affine",
            "PiecewiseAffine",
            "CoarseDropout",
            "KeepSizeByResize",
            "CropToFixedSize",
            "TranslateX",
            "TranslateY",
            "Pad",
            "Lambda",
        ]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        for bs in range(self.batch_size):
            # Store shapes before augmentation to compare
            image_shape = image[
                bs,
            ].shape
            mask_shape = masks[
                bs,
            ].shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = self.augmentation.to_deterministic()
            image[bs,] = det.augment_image(
                image[
                    bs,
                ]
            )

            # for each mask, slow?
            # for c in range(masks[bs, ].shape[-1]):
            # 	uint8_mask = masks[bs, ..., c].astype(np.uint8)

            # 	masks[bs, ..., c] = det.augment_image(
            # 		uint8_mask, hooks=imgaug.HooksImages(activator=hook)
            # 	).astype(np.float32)

            # in one shot
            masks[bs, ...] = det.augment_image(
                masks[bs, ...].astype(np.uint8),
                hooks=imgaug.HooksImages(activator=hook),
            ).astype(np.float32)

            # Verify that shapes didn't change
            assert (
                image[
                    bs,
                ].shape
                == image_shape
            ), "Augmentation shouldn't change image size"
            assert (
                masks[
                    bs,
                ].shape
                == mask_shape
            ), "Augmentation shouldn't change mask size"

        return image, masks

    def __getitem__(self, index):
        if index > max(self._image_ids):
            raise IndexError(
                f"List index out of range. Size of generator: {self.__len__()}"
            )
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self._image_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        images = self.load_image(indexes)
        masks = self.load_mask(indexes)

        # data augmentation
        if self.augmentation:
            images, masks = self.dataAugmentation(images, masks)

        # image normalization
        images = images / 255.0

        return images, masks

    def get_image_by_name(self, imname):
        assert self.batch_size == 1, "Batch size must be 1."
        for i in range(len(self._image_ids)):
            if imname in self.image_reference(i):
                return self.__getitem__(i)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.num_images // self.batch_size
