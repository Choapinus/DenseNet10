# DenseNet10

Andres Valenzuela and Juan Tapia

## Towards an Efficient Segmentation Algorithm for Near-Infrared Eyes Images.

Semantic segmentation has been widely used for several applications, including the detection of eye structures. This is used in tasks such as eye-tracking and gaze estimation, which are useful techniques for human-computer interfaces, salience detection, and Virtual reality (VR), amongst others. Most of the state of the art techniques achieve high accuracy but with a considerable number of parameters. This article explores alternatives to improve the efficiency of the state of the art method, namely DenseNet Tiramisu, when applied to NIR image segmentation. This task is not trivial; the reduction of block and layers also affects the number of feature maps. The growth rate (k) of the feature maps regulates how much new information each layer contributes to the global state, therefore the trade-off amongst grown rate (k), IOU, and the number of layers needs to be carefully studied. The main goal is to achieve a light-weight and efficient network with fewer parameters than traditional architectures in order to be used for mobile device applications. As a result, a DenseNet with only three blocks and ten layers is proposed (DenseNet10). Experiments show that this network achieved higher IOU rates when comparing with Encoder-Decoder, DensetNet56-67-103, MaskRCNN, and DeeplabV3+ models in the Facebook database. Furthermore, this method reached 8th place in The Facebook semantic segmentation challenge with 0.94293 mean IOU and 202.084 parameters with a final score of 0.97147. This score is only 0,001 lower than the first place in the competition. The sclera was identified as the more challenging structure to be segmented.

## Semantic Segmentation of Periocular Near-Infra-Red Eye Images Under Alcohol Effects

This paper proposes a new framework to detect, segment, and estimate the localization of the eyes from a periocular Near-Infra-Red iris image under alcohol consumption. This stage will take part in the final solution to measure the fitness for duty. Fitness systems allow us to determine whether a person is physically or psychologically able to perform their tasks. Our segmentation framework is based on an object detector trained from scratch to detect both eyes from a single image. Then, two efficient networks were used for semantic segmentation; a Criss-Cross attention network and DenseNet10, with only 122,514 and 210,732 parameters, respectively. These networks can find the pupil, iris, and sclera. In the end, the binary output eye mask is used for pupil and iris diameter estimation with high precision. Five state-of-the-art algorithms were used for this purpose. A mixed proposal reached the best results. A second contribution is establishing an alcohol behavior curve to detect the alcohol presence utilizing a stream of images captured from an iris instance. Also, a manually labeled database with more than 20k images was created. Our best method obtains a mean Intersection-over-Union of 94.54% with DenseNet10 with only 210,732 parameters and an error of only 1-pixel on average.


![figure15_paper_alcohol](https://user-images.githubusercontent.com/45126159/174738149-2bc36770-e9ec-4504-a4d2-836c947eed35.png)
<em>Example of semi-closed challenging images. Left original images under alcohol consumption. Right: Our proposed
DenseNet10 results.</em>


### Keras implementation

Developed for [Towards an Efficient Segmentation Algorithm for Near-Infrared Eyes Images](https://ieeexplore.ieee.org/abstract/document/9200989) and complemented in [Semantic Segmentation of Periocular Near-Infra-Red Eye Images Under Alcohol Effects](https://ieeexplore.ieee.org/abstract/document/9502109).

- Train file # input shape must be equal to (320 x 240 x 3)
- Augmentations
- Database
- Env file

### 1. Segmentation stage
-  This process is performed pixel by pixel, assigning a label to each pixel of the input image. By this way, we can separate each region of interest of the eye semantically, obtaining the iris, pupil and sclera in different channels of the output tensor. In the method "predict" from the DenseSegmentator class in [dense10_segmentator.py](https://github.com/Choapinus/DenseNet10/blob/master/dense10_segmentator.py) file we can get this feature as is.

### 2. Pupil and Iris radii estimation stage

- Before the segmentation stage, we can obtain the radii of pupil and iris with the two actual algorythms included in the project (center of mass and MSE2). This methods gives pupil and iris coords as an ellipse [x_init, y_init, r_min, r_max] where x and y are the center of the pupil/iris and the other coords are the min-max radii of the ellipse. This information can be obtained running the "forward" method of the DenseSegmentator class disposed in [dense10_segmentator.py](https://github.com/Choapinus/DenseNet10/blob/master/dense10_segmentator.py). 

<img class="center" src="https://raw.githubusercontent.com/Choapinus/DenseNet10/master/static/radii_estimation.png?token=GHSAT0AAAAAABSXZOPT3NGWCPA5YVLIODU6YY3NYGA">

## Datasets
- OpenEDS is a data set of eye images captured using a virtual-reality HMD with two synchronized eye-facing cameras at a frame rate of 200 Hz under controlled illumination.This dataset is composed of: Semantic segmentation data set collected with 152 participants of 12,759 images with annotations at a resolution of 400×640. Generative data set collected with 152 participants of 252,690 images at a resolution 400×600. Sequence data set collected with 152 participants of 91,200 images at a resolution of 400×640, with duration of 1.5 seconds for each participant, sampled at 200 Hz. Left and right paired human eye corneal topography in the form of point cloud collected for 143 participants. 

Dataset available from the source: https://www.v7labs.com/open-datasets/facebook-openeds

## How to make env and train
```
conda create --file env.yml
conda activate fb_iris
python train.py
```

## How to load and eval
- [Evaluate](https://github.com/Choapinus/DenseNet10/blob/master/notebooks/eval_radio_error.ipynb)
- [Load Abstract Class](https://github.com/Choapinus/DenseNet10/blob/master/notebooks/DenseSegmentatorClass.ipynb)

## Pretrained Models
- [DenseNet10-OpenEDS](https://github.com/Choapinus/DenseNet10/raw/master/models/epoch_124_miou_0.9345.h5)

## How to read coords
- Pupil and iris coords are setted in a ellipse [x_init, y_init, r_min, r_max] where the x and y are the center of the pupil/iris and the rest corresponds to min-max radii of the ellipse.

## How to identify masks
- In the segmentation head, the first channel correspond to background, the second channel correspond to sclera, the third correspond to iris and the last correspond to pupil mask. An example can be found in the [generator file](https://github.com/Choapinus/DenseNet10/blob/dcd9fb5e25a1638f576b7d60efeb3a2fedae0269/utils/datagenerator.py#L213).

## Citation:
```
@ARTICLE{9200989,
  author={Valenzuela, Andres and Arellano, Claudia and Tapia, Juan E.},
  journal={IEEE Access}, 
  title={Towards an Efficient Segmentation Algorithm for Near-Infrared Eyes Images}, 
  year={2020},
  volume={8},
  number={},
  pages={171598-171607},
  doi={10.1109/ACCESS.2020.3025195}}
```
```
@ARTICLE{9502109,
  author={Tapia, Juan E. and Droguett, Enrique Lopez and Valenzuela, Andres and Benalcazar, Daniel P. and Causa, Leonardo and Busch, Christoph},
  journal={IEEE Access}, 
  title={Semantic Segmentation of Periocular Near-Infra-Red Eye Images Under Alcohol Effects}, 
  year={2021},
  volume={9},
  number={},
  pages={109732-109744},
  doi={10.1109/ACCESS.2021.3101256}}
```


## Information
- @Juan Tapia - juan.tapia-farias@h-da.de
- @Andres Valenzuela - andres.valenzuela@tocbiometrics.com


## Disclaimer
The dataset, the implementation, or trained models used is restricted to research purposes. Using the dataset or the implementation/trained models for product development or comercial product is not allowed. This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
