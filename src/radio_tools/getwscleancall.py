#! /usr/bin/python3.9

from astropy.io import fits
import sys

def main():
    filename = sys.argv[1]

    head = fits.getheader(filename)
    hist = head['HISTORY']
    call = ''.join(hist)
    print(call)
