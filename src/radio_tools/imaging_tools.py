import numpy as np

def std_img(data,niter=10,tol=0.001):
    '''
        Determine the stadnard deviation of an image by iteratively masking
        Stops when either niter is reached or the standard deviation changes by less than tol,
        whichever comes first
    '''
    std = np.std(data)
    for i in range(niter):
        mask = data < std
        std_new = np.std(data[mask])
        if np.abs(std_new-std)/std < tol:
            break
        std = std_new
    return std

def calculate_pixels_per_beam(header):
    '''
        Calculate the number of pixels per beam in a FITS header
    '''
    bmaj = header['BMAJ']
    bmin = header['BMIN']
    cdelt = np.abs(header['CDELT1'])
    pix_per_beam = 2*np.pi*bmaj*bmin/(cdelt**2*np.log(2))
    return pix_per_beam