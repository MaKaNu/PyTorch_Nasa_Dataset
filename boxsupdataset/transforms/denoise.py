from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, denoise_tv_bregman)
from skimage import img_as_float


class TotalVariation(object):
    """Denoises the Image with Total Variation Filter
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


class TotalVariation2(object):
    """Denoises the Image with Total Variation Filter
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

    def __getWeight(self):
        return self.__weight

    def __getIsotropic(self):
        return self.__isotropic

    def __getMultichannel(self):
        return self.__multichannel

    def __setWeight(self, x):
        if isinstance(x, float):
            self.__weight = x
        else:
            raise ValueError(x, "weight needs to be a float value")

    def __setIsotropic(self, x):
        if isinstance(x, bool):
            self.__isotropic = x
        else:
            raise ValueError(x, "isotrpoic needs to be a bool value")

    def __setMultichannel(self, x):
        if isinstance(x, bool):
            self.__multichannel = x
        else:
            raise ValueError(x, "multichannel needs to be a bool value")

    weight = property(__getWeight, __setWeight)
    isotropic = property(__getIsotropic, __setIsotropic)
    multichannel = property(__getMultichannel, __setMultichannel)


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
                 sigmaColor: float = 0.05,
                 sigmaSpatial: float = 15.0,
                 multichannel: bool = True) -> None:
        assert False, "THIS TRANSFORMATION IS UNDER CONSTRUCTION!"
        assert isinstance(sigmaColor, float), \
            "sigmaColor needs to be a float value"
        assert isinstance(sigmaSpatial, float), \
            "sigmaSpatial needs to be a float value"
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        self.__sigmaColor = sigmaColor
        self.__sigmaSpatial = sigmaSpatial
        self.__multichannel = multichannel

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_bilateral(image,
                                  sigma_color=self.sigmaColor,
                                  sigma_spatial=self.sigmaSpatial,
                                  multichannel=self.multichannel)

        return {'image': image, 'label': label}

    def __getSigmaColor(self):
        return self.__sigmaColor

    def __getsigmaSpatial(self):
        return self.__sigmaSpatial

    def __getMultichannel(self):
        return self.__multichannel

    def __setSigmaColor(self, x):
        if isinstance(x, float):
            self.__sigmaColor = x
        else:
            raise ValueError(x, "sigmaColor needs to be a float value")

    def __setSigmaSpatial(self, x):
        if isinstance(x, float):
            self.__sigmaSpatial = x
        else:
            raise ValueError(x, "sigmaSpatial needs to be a float value")

    def __setMultichannel(self, x):
        if isinstance(x, bool):
            self.__multichannel = x
        else:
            raise ValueError(x, "multichannel needs to be a bool value")

    sigmaColor = property(__getSigmaColor, __setSigmaColor)
    sigmaSpatial = property(__getsigmaSpatial, __setSigmaSpatial)
    multichannel = property(__getMultichannel, __setMultichannel)


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
                 rescaleSigma: bool = True) -> None:
        assert isinstance(multichannel, bool), \
            "multichannel needs to be a bool value"
        assert isinstance(convert2ycbcr, bool), \
            "convert2ycbr needs to be a bool value"
        assert isinstance(rescaleSigma, bool), \
            "rescaleSigma needs to be a bool value"
        self.__multichannel = multichannel
        self.__convert2ycbcr = convert2ycbcr
        self.__rescaleSigma = rescaleSigma

    def __call__(self, sample: dict) -> dict:
        image, label = sample['image'], sample['label']
        image = img_as_float(image)
        image = denoise_wavelet(image,
                                multichannel=self.multichannel,
                                convert2ycbcr=self.convert2ycbcr,
                                rescale_sigma=self.rescaleSigma)

        return {'image': image, 'label': label}

    def __getMultichannel(self):
        return self.__multichannel

    def __getConvert2ycbcr(self):
        return self.__convert2ycbcr

    def __getRescaleSigma(self):
        return self.__rescaleSigma

    def __setMultichannel(self, x):
        if isinstance(x, bool):
            self.__multichannel = x
        else:
            raise ValueError(x, "multichannel needs to be a bool value")

    def __setConvert2ycbcr(self, x):
        if isinstance(x, bool):
            self.__convert2ycbcr = x
        else:
            raise ValueError(x, "convert2ycbcr needs to be a bool value")

    def __setRescaleSigma(self, x):
        if isinstance(x, bool):
            self.__rescaleSigma = x
        else:
            raise ValueError(x, "rescaleSigma needs to be a bool value")

    multichannel = property(__getMultichannel, __setMultichannel)
    convert2ycbcr = property(__getConvert2ycbcr, __setConvert2ycbcr)
    rescaleSigma = property(__getRescaleSigma, __setRescaleSigma)
