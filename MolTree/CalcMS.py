# Test routine for calculating the isotope distribution.
import numpy as np
import matplotlib.pyplot as plt
import Isotope as iso

# Constuct the molecule.
# A list for element composition.
Elist = ['C', 'H', 'O', 'N']
# A array for related element numbers.
eleN = np.array([9, 4, 4, 0])
eleN2 = np.array([8, 3, 4, 1])
# Extract the isotope information for every elements.
# A list for isotope mass.
isoM = []
isoM2 = []
# A list for isotope abundance.
isoA = []
isoA2 = []
# A list for the number of isotopes.
isoN = []
isoN2 = []
# Scan every elements.
for EName in Elist:
    # Get isotope mass.
    isoM.append(iso.isotope_dict[EName]['Mass'])
    # Get isotope abundance.
    isoA.append(iso.isotope_dict[EName]['Abundance'])
    # Get isotope number.
    isoN.append(iso.isotope_dict[EName]['Num'])

for EName in Elist:
    # Get isotope mass.
    isoM2.append(iso.isotope_dict[EName]['Mass'])
    # Get isotope abundance.
    isoA2.append(iso.isotope_dict[EName]['Abundance'])
    # Get isotope number.
    isoN2.append(iso.isotope_dict[EName]['Num'])

# Calulate the minimum and maximum mz for a specific molecule.
mzMin = 0
mzMax = 0
for idx in range(len(isoM)):
    mzMin += isoM[idx][0] * eleN[idx]
    mzMax += isoM[idx][-1] * eleN[idx]

# Parameters.
# FWHM
FWHM = 0.4
# Maximum mz to be observed.
mzMax = np.ceil(mzMax + 10)
# Minimum mz to be obserbed.
mzMin = np.floor(mzMin - 10)
# Expanded the sampling frequent by k folds.
k = 5
# Precision of the mz axis.
dmz = 0.001
# Tolerance of the Gaussian.
tol = 6

# Call the calculation function.
I, mz = iso.idUniFFT(isoM, isoA, isoN, eleN, FWHM, tol, mzMax, k, dmz)
I2, mz2 = iso.idUniFFT(isoM2, isoA2, isoN2, eleN2, FWHM, tol, mzMax, k, dmz)
I = I + I2
# Find peaks.
Idiff = np.convolve(I, np.array([1, -1]), 'full')
idxPeak = []
for i in range(I.size):
    if Idiff[i] > 0 >= Idiff[i + 1] and I[i] > 0.01:
        idxPeak.append(i)

