from . import imaging_tools
from astropy.io import fits
import argparse

def std_img():
    parser = argparse.ArgumentParser(description='Determine the standard deviation of an image by iteratively masking')
    parser.add_argument('image', type=str, help='Image to calculate standard deviation')
    parser.add_argument('--niter', type=int, default=10, help='Number of iterations to run')
    parser.add_argument('--tol', type=float, default=0.001, help='Tolerance for standard deviation change')
    args = parser.parse_args()

    data = fits.getdata(args.image)
    std = imaging_tools.std_img(data,niter=args.niter,tol=args.tol)
    return std

def calculate_pixels_per_beam():
    parser = argparse.ArgumentParser(description='Calculate the number of pixels per beam in a FITS header')
    parser.add_argument('header', type=str, help='FITS header to calculate pixels per beam')
    args = parser.parse_args()

    header = fits.getheader(args.header)
    pix_per_beam = imaging_tools.calculate_pixels_per_beam(header)
    return pix_per_beam