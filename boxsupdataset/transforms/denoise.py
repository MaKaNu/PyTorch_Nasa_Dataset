""" This Module includes a number of classes which used as transformations
    for pythorch Datasets:
    The module represents Image denoising:
        TotalVariation: Uses a Total Variation Filter after Chambolle
        TotalVariation2: Uses a Total Variation Filter after Bregman
        Bilateral: Uses a bilateral Filter
        Wavelet: Uses a Wavelet Filter
"""

from __future__ import absolute_import
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, denoise_tv_bregman)
from skimage import img_as_float


class TotalVariation(object):
    """Denoises the Image with Total Variation Filter (Chambolle)
    ReadMore:
    https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle

    Args:
        weight (float): Denoising weight. The greater weight, the more
            denoising (at the expense of fidelity to input).
        multichannel (bool): Apply total-variation denoising separately for
            each channel. This option should be true for color images,
            otherwise the denoising is also applied in the channels dimension.
    """
    def __init__(self,
                 weight: float = 0.1,
                 multichannel: bool = True) -> None:
        assert isinstance(weight, float), \
            "weight needs to be a float value"
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        self.__weight = weight
        self.__multichannel = multichannel

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_tv_chambolle(image,
                                     weight=self.weight,
                                     multichannel=self.multichannel)

        return {'image': image, 'label': label}

    def __getWeight__(self):
        return self.__weight

    def __getMultichannel__(self):
        return self.__multichannel

    def __setWeight__(self, var_to_set):
        if isinstance(var_to_set, float):
            self.__weight = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "weight needs to be a float value")

    def __setMultichannel__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__multichannel = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "multichannel needs to be a bool value")

    weight = property(__getWeight__, __setWeight__)
    multichannel = property(__getMultichannel__, __setMultichannel__)


class TotalVariation2(object):
    """Denoises the Image with Total Variation Filter (Bregman)
    ReadMore:
    https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_tv_bregman

    Args:
        weight (float): Denoising weight. The smaller the weight, the more
            denoising (at the expense of less similarity to the input). The
            regularization parameter lambda is chosen as 2 * weight.
        isotropic (bool): Switch between isotropic and anisotropic TV
            denoising.
        multichannel (bool): Apply total-variation denoising separately for
            each channel. This option should be true for color images,
            otherwise the denoising is also applied in the channels dimension.
    """
    def __init__(self,
                 weight: float = 4.0,
                 isotropic: bool = True,
                 multichannel: bool = True) -> None:
        assert isinstance(weight, float), \
            "weight needs to be a float value"
        assert isinstance(isotropic, bool), \
            "isotropic needs to be a bool value"
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        self.__weight = weight
        self.__isotropic = isotropic
        self.__multichannel = multichannel

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_tv_bregman(image,
                                   weight=self.weight,
                                   isotropic=self.isotropic,
                                   multichannel=self.multichannel)

        return {'image': image, 'label': label}

    def __getWeight__(self):
        return self.__weight

    def __getIsotropic__(self):
        return self.__isotropic

    def __getMultichannel__(self):
        return self.__multichannel

    def __setWeight__(self, var_to_set):
        if isinstance(var_to_set, float):
            self.__weight = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "weight needs to be a float value")

    def __setIsotropic__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__isotropic = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "isotrpoic needs to be a bool value")

    def __setMultichannel__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__multichannel = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "multichannel needs to be a bool value")

    weight = property(__getWeight__, __setWeight__)
    isotropic = property(__getIsotropic__, __setIsotropic__)
    multichannel = property(__getMultichannel__, __setMultichannel__)


class Bilateral(object):
    """Denoises the Image with bilateral Filter
    ReadMore:
    https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_bilateral

    Args:
        sigmaColor (float): Standard deviation for grayvalue/color distance
            (radiometric similarity). A larger value results in averaging of
            pixels with larger radiometric differences. Note, that the image
            will be converted using the img_as_float function and thus the
            standard deviation is in respect to the range [0, 1]. If the value
            is None the standard deviation of the image will be used.
        sigmaSpatial (float): Standard deviation for range distance. A larger
            value results in averaging of pixels with larger spatial
            differences.
        multichannel (bool): Whether the last axis of the image is to be
            interpreted as multiple channels or another spatial dimension.
    """
    def __init__(self,
                 sigma_color: float = 0.05,
                 sigma_spatial: float = 15.0,
                 multichannel: bool = True) -> None:
        assert False, "THIS TRANSFORMATION IS UNDER CONSTRUCTION!"
        assert isinstance(sigma_color, float), \
            "sigmaColor needs to be a float value"
        assert isinstance(sigma_spatial, float), \
            "sigmaSpatial needs to be a float value"
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        self.__sigma_color = sigma_color
        self.__sigma_spatial = sigma_spatial
        self.__multichannel = multichannel

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_bilateral(image,
                                  sigma_color=self.sigma_color,
                                  sigma_spatial=self.sigma_spatial,
                                  multichannel=self.multichannel)

        return {'image': image, 'label': label}

    def __getSigmacolor__(self):
        return self.__sigma_color

    def __getsigmaspatial__(self):
        return self.__sigma_spatial

    def __getMultichannel__(self):
        return self.__multichannel

    def __setSigmacolor__(self, var_to_set):
        if isinstance(var_to_set, float):
            self.__sigma_color = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "sigmaColor needs to be a float value")

    def __setSigmaspatial__(self, var_to_set):
        if isinstance(var_to_set, float):
            self.__sigma_spatial = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "sigmaSpatial needs to be a float value")

    def __setMultichannel__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__multichannel = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "multichannel needs to be a bool value")

    sigma_color = property(__getSigmacolor__, __setSigmacolor__)
    sigma_spatial = property(__getsigmaspatial__, __setSigmaspatial__)
    multichannel = property(__getMultichannel__, __setMultichannel__)


class Wavelet(object):
    """Denoises the Image with wavelet.
    ReadMore:
    https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.denoise_wavelet

    Args:
        multichannel (bool): Apply total-variation denoising separately for
            each channel. This option should be true for color images,
            otherwise the denoising is also applied in the channels dimension.
        convert2ycbcr (bool): If True and multichannel True, do the wavelet
            denoising in the YCbCr colorspace instead of the RGB color space.
            This typically results in better performance for RGB images.
        rescaleSigma (bool): If False, no rescaling of the user-provided sigma
            will be performed. The default of None rescales sigma appropriately
            if the image is rescaled internally. A DeprecationWarning is raised
            to warn the user about this new behaviour. This warning can be
            avoided by setting rescale_sigma=True.
    """
    def __init__(self,
                 multichannel: bool = True,
                 convert2ycbcr: bool = False,
                 rescale_sigma: bool = True) -> None:
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        assert isinstance(convert2ycbcr, bool), \
            "convert2ycbr needs to be a bool value"
        assert isinstance(rescale_sigma, bool), \
            "rescale_sigma needs to be a bool value"
        self.__multichannel = multichannel
        self.__convert2ycbcr = convert2ycbcr
        self.__rescale_sigma = rescale_sigma

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_wavelet(image,
                                multichannel=self.multichannel,
                                convert2ycbcr=self.convert2ycbcr,
                                rescale_sigma=self.rescale_sigma)

        return {'image': image, 'label': label}

    def __getMultichannel__(self):
        return self.__multichannel

    def __getConvert2ycbcr__(self):
        return self.__convert2ycbcr

    def __getRescaleSigma__(self):
        return self.__rescale_sigma

    def __setMultichannel__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__multichannel = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "multichannel needs to be a bool value")

    def __setConvert2ycbcr__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__convert2ycbcr = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "convert2ycbcr needs to be a bool value")

    def __setRescaleSigma__(self, var_to_set):
        if isinstance(var_to_set, bool):
            self.__rescale_sigma = var_to_set
        else:
            raise ValueError(
                var_to_set,
                "rescaleSigma needs to be a bool value")

    multichannel = property(__getMultichannel__, __setMultichannel__)
    convert2ycbcr = property(__getConvert2ycbcr__, __setConvert2ycbcr__)
    rescale_sigma = property(__getRescaleSigma__, __setRescaleSigma__)
