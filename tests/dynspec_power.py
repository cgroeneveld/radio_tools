from radio_tools import dynspec
from radio_tools import lib_dyn
import numpy as np

# Load the spectrum from file
x, y, yerr = np.loadtxt('spec_3c196.txt', unpack=True)

ensemb = dynspec.MultiPowerlaw([[1e-6,1e3], [-3, 0],[-3,0]], x, y, yerr)
best_model = ensemb.run()

print(best_model(45e6))