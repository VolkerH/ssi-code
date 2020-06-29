import sys
import time
from pathlib import Path
import numpy
from imageio import imread
from tifffile import imwrite
import napari
from pathlib import Path

from ssi.ssi_deconv import SSIDeconvolution
from ssi.lr_deconv import ImageTranslatorLRDeconv
from ssi.models.unet import UNet
from ssi.utils.io.datasets import normalise, add_microscope_blur_2d, add_poisson_gaussian_noise
from ssi.utils.metrics.image_metrics import psnr, spectral_mutual_information, mutual_information, ssim


generic_2d_mono_raw_folder = Path("../benchmark/images/generic_2d_all")


def get_benchmark_image(type, name):
    folder = generic_2d_mono_raw_folder / type
    files = [f for f in folder.iterdir() if f.is_file()]
    filename = [f.name for f in files if name in f.name][0]
    filepath = folder / filename
    array = imread(filepath)
    return array, filename



def printscore(header, val1, val2, val3, val4):
    print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")


def demo(image_clipped):
    image_clipped = normalise(image_clipped.astype(numpy.float32))
    blurred_image, psf_kernel = add_microscope_blur_2d(image_clipped)
    # noisy_blurred_image = add_noise(blurred_image, intensity=None, variance=0.01, sap=0.01, clip=True)
    noisy_blurred_image = add_poisson_gaussian_noise(blurred_image, alpha=0.001, sigma=0.1, sap=0.01, quant_bits=10)


    for i in range(10):
        it_deconv = SSIDeconvolution(
            max_epochs=3000,
            patience=300,
            batch_size=8,
            learning_rate=0.01,
            normaliser_type='identity',
            psf_kernel=psf_kernel,
            model_class=UNet,
            masking=True,
            masking_density=0.01,
            loss='l2',
        )

        start = time.time()
        it_deconv.train(noisy_blurred_image)
        stop = time.time()
        print(f"Training: elapsed time:  {stop - start} ")

        start = time.time()
        deconvolved_image = it_deconv.translate(noisy_blurred_image)
        stop = time.time()
        print(f"inference: elapsed time:  {stop - start} ")

        image_clipped = numpy.clip(image_clipped, 0, 1)
        deconvolved_image_clipped = numpy.clip(deconvolved_image, 0, 1)


        printscore(
            "ssi deconv            : ",
            psnr(image_clipped, deconvolved_image_clipped),
            spectral_mutual_information(image_clipped, deconvolved_image_clipped),
            mutual_information(image_clipped, deconvolved_image_clipped),
            ssim(image_clipped, deconvolved_image_clipped),
        )

        print("NOTE: if you get a bad results for ssi, blame stochastic optimisation and retry...")
        print("      The training is done on the same exact image that we infer on, very few pixels...")
        print("      Training should be more stable given more data...")

        folder = Path(".") / f"{str(i)}"
        folder.mkdir(exist_ok=True)

        #imwrite(str(fo'image2d.tif',image)
        #imwrite('blurred2d.tif', blurred_image)
        imwrite(str(folder/ 'noisyblurred.tif'),noisy_blurred_image)
        #imwrite('lr_deconvolved_image_2.tif',lr_deconvolved_image_2_clipped)
        #imwrite('lr_deconvolved_image_5.tif',lr_deconvolved_image_5_clipped)
        #imwrite('lr_deconvolved_image_10.tif',lr_deconvolved_image_10_clipped)
        #imwrite('lr_deconvolved_image_20.tif',lr_deconvolved_image_20_clipped)
        imwrite(str(folder/'ssi_deconvolved_clipped_image.tif'), deconvolved_image_clipped)
        imwrite(str(folder/'ssi_deconvolved_image.tif'), deconvolved_image)




if __name__ == '__main__':
    image_name = 'drosophila'
    if len(sys.argv) > 1:
        image_name = sys.argv[1].rstrip().lstrip()
    image, _ = get_benchmark_image('gt', image_name)
    demo(image)
