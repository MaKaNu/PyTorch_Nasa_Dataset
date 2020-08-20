from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float


class TotalVariation(object):
    """Denoises the Image with Total Variation Filter
    ReadMore:
    https://scikit-image.org/docs/dev/api/skimage.restoration.html?highlight=denoise_tv_chambolle#skimage.restoration.denoise_tv_chambolle # noqa

    Args:
        weight (float): Denoising weight. The greater weight, the more
            denoising (at the expense of fidelity to input).
        multichannel (bool): Apply total-variation denoising separately for
            each channel. This option should be true for color images,
            otherwise the denoising is also applied in the channels dimension.
    """
    def __init__(self, weight: float = 0.1, multichannel: bool = True) -> None:
        assert isinstance(weight, float), \
            "weight needs to be a float value"
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        self.__weight = weight
        self.__multichannel = multichannel

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_tv_chambolle(image, self.weight, self.multichannel)

        return {'image': image, 'label': label}

    def __getWeight(self):
        return self.__weight

    def __getMultichannel(self):
        return self.__multichannel

    def __setWeight(self, x):
        if isinstance(x, float):
            self.__weight = x
        else:
            raise ValueError(x, "weight needs to be a float value")

    def __setMultichannel(self, x):
        if isinstance(x, bool):
            self.__multichannel = x
        else:
            raise ValueError(x, "multichannel needs to be a bool value")

    weight = property(__getWeight, __setWeight)
    multichannel = property(__getMultichannel, __setMultichannel)