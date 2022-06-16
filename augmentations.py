from imgaug import (
    augmenters as iaa,
)  # https://github.com/aleju/imgaug (pip3 install imgaug)


def avg_aug(mode="constant"):

    # define mode. ('constant', None)
    mode = "constant"

    # sequential augmentation
    aug = iaa.SomeOf(
        (0, 12),
        [
            iaa.Sometimes(0.5, iaa.Fliplr(1)),
            iaa.Sometimes(0.5, iaa.Flipud(1)),
            iaa.Sometimes(
                0.5,
                [
                    iaa.Affine(rotate=(-30, 30), mode=mode),
                ],
            ),
            iaa.Sometimes(
                0.5,
                [
                    iaa.Affine(
                        scale=(0.75, 1.5),
                        mode=mode,
                        translate_percent={
                            "x": (-0.5, 0.5),
                            "y": (-0.5, 0.5),
                        },
                    ),
                ],
            ),
            iaa.Sometimes(
                0.35,
                [
                    iaa.CoarseDropout(p=(0.05, 0.25)),
                ],
            ),
            iaa.Add(value=(-30, 30)),
            iaa.SomeOf(
                (0, 2),
                [
                    iaa.AdditiveGaussianNoise(loc=(0.01 * 255, 0.1 * 255)),
                    iaa.AdditiveLaplaceNoise(loc=(0.01 * 255, 0.1 * 255)),
                    iaa.AdditivePoissonNoise(lam=(0.01, 16.0)),
                ],
            ),
            iaa.SomeOf(
                (0, 2),
                [
                    iaa.GaussianBlur(sigma=(0.01, 2.5)),
                    iaa.AverageBlur(k=(1, 5)),
                    iaa.MotionBlur(k=(3, 5)),
                    iaa.GammaContrast(gamma=(0.01, 1.0)),
                    iaa.imgcorruptlike.GaussianBlur(severity=(1, 3)),
                    iaa.imgcorruptlike.GlassBlur(severity=(1, 3)),  # tambien se demora
                    iaa.imgcorruptlike.DefocusBlur(severity=(1, 3)),
                    iaa.imgcorruptlike.MotionBlur(severity=(1, 3)),  # se demora a cagar
                    iaa.imgcorruptlike.ZoomBlur(severity=(1, 3)),  # se demora a cagar
                ],
            ),
            iaa.SomeOf(
                (0, 1),
                [
                    iaa.imgcorruptlike.Frost(severity=(1, 2)),
                    iaa.imgcorruptlike.Spatter(severity=(1, 2)),
                    iaa.Snowflakes(flake_size=(0.01, 0.2), speed=(0.01, 0.05)),
                    iaa.Rain(speed=(0.01, 0.1)),
                ],
            ),
            iaa.SomeOf(
                (0, 2),
                [
                    iaa.PerspectiveTransform(scale=(0.01, 0.2), keep_size=True),
                    iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # se demora mucho
                    iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.25),
                ],
            ),
            iaa.Sometimes(0.5, iaa.JpegCompression(compression=(90, 98))),
            # add random state to augmenter
        ],
        random_state=True,
    )

    return aug
