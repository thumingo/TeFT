# Data and functions for calculating the 
# ideal isotope distributions for mass spectrometry applications.
import numpy as np

##
# An dictionary for isotope mass and abundance,
# extracted from https://ciaaw.org/ .
##
isotope_dict = {'H': {'Num': int(2),
                      'Mass': np.array([1.007825, 2.0141018]),
                      'Abundance': np.array([0.999985, 0.000015])},
                # Hydrogen, 2022/10/22

                ##
                'C': {'Num': int(2),
                      'Mass': np.array([12, 13.003355]),
                      'Abundance': np.array([0.9894, 0.0106])},
                # Carbon, 2022/10/22

                ##
                'N': {'Num': int(2),
                      'Mass': np.array([14.003074, 15.000109]),
                      'Abundance': np.array([0.996205, 0.003795])},
                # Nitrogen, 2022/10/22

                ##
                'O': {'Num': int(3),
                      'Mass': np.array([15.994915, 16.999132, 17.99916]),
                      'Abundance': np.array([0.99757, 0.000385, 0.002045])},
                # Oxygen, 2022/10/22

                ##
                'F': {'Num': int(1),
                      'Mass': np.array([18.998403]),
                      'Abundance': np.array([1])},
                # Fluorine, 2022/10/22

                ##
                'Na': {'Num': int(1),
                       'Mass': np.array([22.989769]),
                       'Abundance': np.array([1])},
                # Sodium, 2022/10/22

                ##
                'Mg': {'Num': int(3),
                       'Mass': np.array([23.985042, 24.985837, 25.982593]),
                       'Abundance': np.array([0.78964, 0.10011, 0.11025])},
                # Magnesium, 2022/10/22

                ##
                'Al': {'Num': int(1),
                       'Mass': np.array([26.981538]),
                       'Abundance': np.array([1])},
                # Aluminium, 2022/10/22

                ##
                'Si': {'Num': int(3),
                       'Mass': np.array([27.976927, 28.976495, 29.97377]),
                       'Abundance': np.array([0.922545, 0.04672, 0.030735])},
                # Silicon, 2022/10/22

                ##
                'Cl': {'Num': int(2),
                       'Mass': np.array([34.968853, 36.965903]),
                       'Abundance': np.array([0.758, 0.242])},
                # Chlorine, 2022/10/22

                ##
                'K': {'Num': int(3),
                      'Mass': np.array([38.963706, 39.963998, 40.961825]),
                      'Abundance': np.array([0.932581, 0.000117, 0.067302])},
                # Potassium, 2022/10/22

                ##
                'Ca': {'Num': int(6),
                       'Mass': np.array([39.962591, 41.958618, 42.958766, 43.955482, 45.95369, 47.952523]),
                       'Abundance': np.array([0.96941, 0.00647, 0.00135, 0.02086, 0.00004, 0.00187])},
                # Calcium, 2022/10/22

                ##
                'P': {'Num': int(1),
                      'Mass': np.array([30.973762]),
                      'Abundance': np.array([1])},
                # Phosphorus, 2022/10/22

                ##
                'S': {'Num': int(4),
                      'Mass': np.array([31.972071, 32.971459, 33.967867, 35.967081]),
                      'Abundance': np.array([0.9485, 0.00763, 0.04365, 0.000158])}
                # Sulfur, 2022/10/22
                }


##


##
# Function for unilateral gaussian signal.
# Inputs:
#          sigma,  parameter for guassian signal;
#          dx,     sampling interval of the independent variable;
#          tol,    tolerance for Gaussian function,
#                  the maximum of the independent variable should be tol*sigma
#          method, 'Original'(default), calculate the signal according to the 
#                  original independent variable;
#                  'Transform', calculate the signal according to the Fourier
#                  domain.
# Outputs:
#          val,    Gaussian signal;
#          x,      related independent variable (time domain or frequency domain)
##
def UniGaussian(sigma, dx, tol, method='Original'):
    # Calculate the signal according to the original independent variable.
    if method == 'Original':
        # Calclate the length of the signal.
        L = np.ceil(tol * sigma / dx)
        # Independent variable.
        x = np.arange(0, L) * dx
        # Gaussian signal.
        val = np.exp(-(x * x) / (2 * sigma * sigma))
        # Adjust the amplitude to ensure the integral equals 1.
        val = val / (sigma * np.sqrt(2 * np.pi))
    ##

    # Calculate the signal according to the Fourier domain.
    if method == 'Transform':
        # Calclate the length of the signal.
        L = np.ceil(tol / (dx * sigma))
        # Independent variable.
        x = np.arange(0, L) * dx
        # When calculating the signal in Fourier domain,
        # the independent variable should multiply 2*pi
        omega = 2 * np.pi * x
        # Gaussian signal.
        # TIPS: theoretical calculation according to the Fourier Transform 
        # using the same parameter of original domain,
        # the amplitude should not be adjusted.
        val = np.exp(- (omega * omega) * (sigma * sigma) / 2)

    # Outputs
    return val, x


##


##
# Function for calculating the isotope distribution of a given molecule 
# by unilateral FFT.
##
def idUniFFT(isoM, isoA, isoN, eleN, FWHM, tol, mzMax, k, dmz):
    # Calculate the sigma parameter for Gaussian
    # according to Full Width Half Maximum.
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))

    # mz axis ('frequecy' domain)
    # To keep consistent with Digital Signal Proccessing,
    # we use 'f' to represent the frequiecy domain.
    # TIPS: for sampling the sinusoidal signal, fs = 2*fmax is not enough;
    # we expand the frequency axis as mzMax*k.
    fmax = mzMax * k
    fs = 2 * fmax
    # Unilateral frequency axis.
    f = np.arange(0, fmax, dmz)
    # Bilateral full length
    sigL = 2 * f.size - 1
    # mz axis, only 0-mzMax is needed.
    mz = np.arange(0, mzMax, dmz)
    mzL = mz.size

    # 'time' domain
    # Sampling interval.
    dt = 1 / fs
    # Calculate unilateral Gaussian as 'Transform' domain.
    g, t = UniGaussian(sigma, dt, tol, method='Transform')
    # Effective length in time domain
    uniL = t.size

    # Generate complex sinusoidal signal for a specific molecule.
    # Initiate the signal.
    sig = np.ones((uniL,), dtype=np.complex128)
    # Scan every element in the molecule.
    for idx1 in range(len(isoM)):
        # Initiate the signal for current element.
        Esig = np.zeros((uniL,), dtype=np.complex128)
        # Scan every isotope for current element.
        for idx2 in range(isoN[idx1]):
            # Accumulation the complex sinusoidal signal.
            Esig += isoA[idx1][idx2] * np.exp(1j * 2 * np.pi * isoM[idx1][idx2] * t)
        # Multiplicative of every elements.
        sig *= Esig ** eleN[idx1]
    # multiply the Gaussian peak.
    sig *= g
    ##

    # Transform.
    # The 'time' signal is a complex sinusoidal signal,
    # the real part is symmetric and the image part is anti-symmetric.
    # thus, we can calculate its unilateral transform,
    # according to the Digital Signal Processing course.
    # Transform the real part using full length sigL
    I1 = np.fft.fft(np.real(sig), sigL)
    # Transform the iamge part using full length sigL
    I2 = np.fft.fft(np.imag(sig), sigL)
    # Construct the overall unilateral transform, only 0-mzMax needed.
    I = 2 * (np.real(I1[0:mzL]) - np.imag(I2[0:mzL])) - np.real(sig[0])
    # Normalization.
    I = I / np.max(I)

    # Outputs
    return I, mz
##
