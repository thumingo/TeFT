from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import xlrd
from scipy import interpolate
from skimage import restoration
from MolTree import Isotope as iso
import csv


def index_number(li, defaultnumber):
    select = defaultnumber - li[0]
    index = 0
    for i in range(1, len(li) - 1):
        select2 = defaultnumber - li[i]
        if abs(select) > abs(select2):
            select = select2
            index = i
    return index


def remove_non_increasing_pairs(X, Y):
    i = 0
    while i < len(X) - 1:
        if X[i] >= X[i + 1]:
            X = np.delete(X, i + 1)
            Y = np.delete(Y, i + 1)
        else:
            i += 1
    return X, Y



# # 读取质谱数据 Kryin3
dai_mz = []
dai_amp = []
with open(r'C:\Users\WIN10\Desktop\小质谱碎裂谱图\4.17碎裂校正2.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        dai_mz.append(float(row[0]))
        dai_amp.append(float(row[1]))
f.close()
mzMin = 250
mzMax = 300
mzMin_num = index_number(dai_mz, mzMin)
mzMax_num = index_number(dai_mz, mzMax)
dai_mz = dai_mz[mzMin_num:mzMax_num]
dai_amp = dai_amp[mzMin_num:mzMax_num]
dai_mz, dai_amp = remove_non_increasing_pairs(dai_mz, dai_amp)
Elist = ['C', 'H', 'O', 'N', 'S']
# A array for related element numbers.
eleN = np.array([15, 11, 4, 0, 0])

isoM = []
# A list for isotope abundance.
isoA = []
# A list for the number of isotopes.
isoN = []
# Scan every elements.
for EName in Elist:
    # Get isotope mass.
    isoM.append(iso.isotope_dict[EName]['Mass'])
    # Get isotope abundance.
    isoA.append(iso.isotope_dict[EName]['Abundance'])
    # Get isotope number.
    isoN.append(iso.isotope_dict[EName]['Num'])

# Calulate the minimum and maximum mz for a specific molecule.

# for idx in range(len(isoM)):
#     mzMin += isoM[idx][0] * eleN[idx]
#     mzMax += isoM[idx][-1] * eleN[idx]

# Parameters.
# FWHM
FWHM = 0.005
# Maximum mz to be observed.
# mzMax = np.ceil(mzMax + 10)
# # Minimum mz to be obserbed.
# mzMin = np.floor(mzMin - 10)
# Expanded the sampling frequent by k folds.
k = 5
# Precision of the mz axis.
dmz = 0.0001
# Tolerance of the Gaussian.
tol = 6

# Call the calculation function.
I, mz = iso.idUniFFT(isoM, isoA, isoN, eleN, FWHM, tol, mzMax, k, dmz)

# Find peaks.

dai_mz = np.array(dai_mz)
dai_amp = np.array(dai_amp)
yy = interpolate.splrep(dai_mz, dai_amp)
chazhi = interpolate.splev(mz[mz >= mzMin], yy, der=0)

# plt.figure(figsize=(24, 12))
# plt.plot(mz[mz > mzMin], I[mz > mzMin], linewidth=2)
# plt.plot(mz[idxPeak], I[idxPeak], 'r*')

K = restoration.richardson_lucy(I[mz >= mzMin], chazhi, num_iter=20)
# np.save("../300-350-Kyrin3-230K-4.17.npy", K)
np.save("../4.17-2.npy", K)
# K= np.load("300-350.npy")
# mzjz = np.array(dai_mz)
# ampjz = np.array(dai_amp)
# ampjz = ampjz / max(ampjz)
# ampjz = ampjz[1800:2880]
# mzjz = mzjz[1800:2880]
# yy = interpolate.splrep(mzjz, ampjz)
# chazhishan = interpolate.splev(mz, yy, der=0)
ans2 = np.convolve(chazhi, K, 'same')
ans2 = ans2 / max(ans2)
dai_amp = dai_amp / max(dai_amp)
mz = mz[mz >= mzMin]
plt.plot(mz, ans2)
Idiff = np.convolve(ans2, np.array([1, -1]), 'full')
idxPeak = []
for i in range(ans2.size):
    if Idiff[i] > 0 >= Idiff[i + 1] and ans2[i] > 0.01:
        idxPeak.append(i)
plt.plot(dai_mz, dai_amp)
plt.show()

