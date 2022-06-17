# DenseNet10
## UNet Segmentator based on Tiramisu DenseNet arch
### Keras version

Created in [Towards an Efficient Segmentation Algorithm for Near-Infrared Eyes Images](https://ieeexplore.ieee.org/abstract/document/9200989) and complemented in [Semantic Segmentation of Periocular Near-Infra-Red Eye Images Under Alcohol Effects](https://ieeexplore.ieee.org/abstract/document/9502109).

- Train file
- Augmentations
- Database
- Env file

## Abstract Towards an Efficient Segmentation (...)

Semantic segmentation has been widely used for several applications, including the detection of eye structures. This is used in tasks such as eye-tracking and gaze estimation, which are useful techniques for human-computer interfaces, salience detection, and Virtual reality (VR), amongst others. Most of the state of the art techniques achieve high accuracy but with a considerable number of parameters. This article explores alternatives to improve the efficiency of the state of the art method, namely DenseNet Tiramisu, when applied to NIR image segmentation. This task is not trivial; the reduction of block and layers also affects the number of feature maps. The growth rate (k) of the feature maps regulates how much new information each layer contributes to the global state, therefore the trade-off amongst grown rate (k), IOU, and the number of layers needs to be carefully studied. The main goal is to achieve a light-weight and efficient network with fewer parameters than traditional architectures in order to be used for mobile device applications. As a result, a DenseNet with only three blocks and ten layers is proposed (DenseNet10). Experiments show that this network achieved higher IOU rates when comparing with Encoder-Decoder, DensetNet56-67-103, MaskRCNN, and DeeplabV3+ models in the Facebook database. Furthermore, this method reached 8th place in The Facebook semantic segmentation challenge with 0.94293 mean IOU and 202.084 parameters with a final score of 0.97147. This score is only 0,001 lower than the first place in the competition. The sclera was identified as the more challenging structure to be segmented.


![](https://raw.githubusercontent.com/Choapinus/DenseNet10/master/static/model_weights.png?token=GHSAT0AAAAAABVWFPF4NULDH2EENONLFJB2YVMYPXA)


## Abstract Segmentation (...) Under Alcohol Effects

This paper proposes a new framework to detect, segment, and estimate the localization of the eyes from a periocular Near-Infra-Red iris image under alcohol consumption. This stage will take part in the final solution to measure the fitness for duty. Fitness systems allow us to determine whether a person is physically or psychologically able to perform their tasks. Our segmentation framework is based on an object detector trained from scratch to detect both eyes from a single image. Then, two efficient networks were used for semantic segmentation; a Criss-Cross attention network and DenseNet10, with only 122,514 and 210,732 parameters, respectively. These networks can find the pupil, iris, and sclera. In the end, the binary output eye mask is used for pupil and iris diameter estimation with high precision. Five state-of-the-art algorithms were used for this purpose. A mixed proposal reached the best results. A second contribution is establishing an alcohol behavior curve to detect the alcohol presence utilizing a stream of images captured from an iris instance. Also, a manually labeled database with more than 20k images was created. Our best method obtains a mean Intersection-over-Union of 94.54% with DenseNet10 with only 210,732 parameters and an error of only 1-pixel on average.


![](https://raw.githubusercontent.com/Choapinus/DenseNet10/master/static/radii_estimation.png?token=GHSAT0AAAAAABVWFPF4ML6OFVAKCZGZBFO4YVMYQIQ)
![](https://raw.githubusercontent.com/Choapinus/DenseNet10/master/static/eyes_segmented.png?token=GHSAT0AAAAAABVWFPF4C7OTXJDGFUC7VOAGYVMYVLQ)

## Datasets
- OpenEDS: coming soon!

## Models
- Dense10 - OpenEDS: coming soon!

## Information
- [Juan Tapia](juan.tapia-farias@h-da.de) 
- [Andres Valenzuela](andres.valenzuela@tocbiometrics.com)
