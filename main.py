# Summary
# Author: Ioan-Alexandru Stancioiu (I.Stancioiu@liverpool.ac.uk)
# Date: 17 / 03 / 2026

# CAVEAT LECTOR: To make things easier for the author any "power of" will be gracefully attached to the variable it is
#                acting on
#########################################
########## IMPORTING LIBRARIES ##########
#########################################
import numpy as np
from scipy.special import jv, hankel1
import matplotlib.pyplot as plt
import time
##################################################################################
########## SETTING UP MATHEMATICAL CONSTANTS AND PHYSICAL DOMAIN ARRAYS ##########
##################################################################################
# Resolving any complex result issues
def sqrt_safe(x):
    return np.sqrt(x + 0j)

# Calculating the wavenumber ratio from Poisson's ratio. This is the equivalent of xi.
def calculate_ratio(v):
    numerator = 2 * (1 - v)
    denominator = 1 - 2 * v
    return np.sqrt(numerator / denominator)

# Setting up the constants
NU = 0.25                                             # Poisson's ratio nu (to be chosen by user)
WAVENUMBER_RATIO = calculate_ratio(NU)                # xi
FREQUENCY = 20                                        # Frequency f_0 (to be chosen by user)
OMEGA = 2 * np.pi * FREQUENCY                         # Angular Frequency
CP = 2000                                             # Compressional Wavespeed (to be chosen by user)
RHO = 2000                                            # Density of continuum (to be chosen by user)
P0 = 500                                              # "Source Strength" (to be chosen by user)
a = 0.05                                              # Radius of disk (to be chosen by user)
ETA = 0.00                                            # Loss Factor (to be chosen by user): 0.00 - lossless environment
                                                      #                                     0.01 - small damping for soil
                                                      #                                     0.03 - large damping for soil
kc = OMEGA / CP                                       # Compressional wavenumber
ks = WAVENUMBER_RATIO * kc                            # Shear wavenumber

MU = RHO * (CP / WAVENUMBER_RATIO)**2                 # Lamé Parameter - mu
C = RHO * CP**2                                       # Lamé Parameters - C = lambda + 2 * mu
LAMBDA = C - 2 * MU                                   # Lamé Parameter - lambda

CR = (OMEGA / ks) * ((0.87 + 1.12 * NU) / (1 + NU))   # Rayleigh wave speed
LAMBDA_R = CR / FREQUENCY                             # Rayleigh wavelength
kR = OMEGA / CR                                       # Rayleigh wave number

# Physical domain parameters
R_MIN = 1e-10
R_MAX = 1000
R_POINTS = 100
Z_SCALE = 3

# Arrays
r = np.linspace(R_MIN, R_MAX, R_POINTS)
z = np.linspace(Z_SCALE * R_MIN, Z_SCALE * R_MAX, Z_SCALE * R_POINTS) # NOTE: Z_SCALE ensures that we integrate beyond
                                                                      #       the semi-sphere to fully encapsulate the
                                                                      #       depth reaching infinity for the full field
                                                                      #       Additionally, 1e-10 * 3 is approximately 0
                                                                      #       from a numerical perspective
dz = z[1] - z[0]

# Dynamically calculate the Rayleigh pole each time we change a parameter.
def find_poles():
    cubic_coeff = [(16 * MU**2 * kc**2 - 16 * MU * C * kc**2), (4 * MU**2 * ks**4 + 16 * MU * C * kc**2 * ks**2 + 4 * C**2 * kc**4 - 16 * MU**2 * kc**2 * ks**2),
               (-4 * MU * C * kc**2 * ks**4 - 4 * C**2 * kc**4 * ks**2), (C**2 * kc**4 * ks**4)]
    X = np.roots(cubic_coeff)             # poles^2
    condition = X[np.imag >= 0]           # Sommerfeld Radiation Condition
    zeta = sqrt_safe(condition)           # poles only one of which is the Rayleigh pole
    rayleigh_pole = zeta[zeta**2 > ks**2] # Isolate just the Rayleigh pole

    return rayleigh_pole

POLE = find_poles().astype(complex)   # returns an array
POLE = POLE[0]                        # return just a complex number

##############################################################
############### INTEGRATION FUNCTION'S KERNELS ###############
##############################################################
def curly_phi(k_r):
    numerator = P0 * a * jv(1, k_r * a)                                                       # Might need (2 * np.pi) and/or delta(w - w0)
    denominator_term_1 = (ks**2 - 2 * k_r**2) * ((LAMBDA + 2 * MU) * kc**2 - 2 * MU * k_r**2)
    denominator_term_2 = - 4 * MU * k_r**2 * sqrt_safe(k_r**2 - ks**2) * sqrt_safe(k_r**2 - kc**2)
    return numerator / (denominator_term_1 + denominator_term_2)

def phi_kernel(k_r):
    numerator = (ks**2 - 2 * k_r**2) * curly_phi(k_r)
    denominator = k_r
    return numerator / denominator

def Phi_kernel(k_r):
    return 2 * sqrt_safe(k_r**2 - kc**2) * curly_phi(k_r)

def exponential_term(sqrt_term):
    return np.exp(-z[np.newaxis,:] * sqrt_term[:, np.newaxis])

def bessel_term(order, k_r):
    return jv(order, k_r[:, np.newaxis] * r[np.newaxis, :])

def comp_u_r_kernel(k_r):
    return (k_r**2 * phi_kernel(k_r))[:, np.newaxis]

def shear_u_r_kernel(k_r):
    return (sqrt_safe(k_r**2 - ks**2) * k_r * Phi_kernel(k_r))[:, np.newaxis]

def comp_u_z_kernel(k_r):
    return (k_r * sqrt_safe(k_r**2 - kc**2) * phi_kernel(k_r))[:, np.newaxis]

def shear_u_z_kernel(k_r):
    return (k_r**2 * Phi_kernel(k_r))[:, np.newaxis]
#################################################
############### INTEGRATION ARRAY ###############
#################################################
DELTA = 1e-4 * kc
ALPHA = kc.real
KR_MAX = 100 * kc.real

POLE_MIN = 0.95 * POLE
POLE_MAX = 1.20 * POLE

PRE_POLE_POINTS = 10_000
POLE_REGION_POINTS = 50_000
N_TAIL = 100_000

kr_1 = np.linspace(0, POLE_MIN, R_POINTS) + 1j * DELTA
kr_2 = np.linspace(POLE_MIN, POLE_MAX, R_POINTS) + 1j * DELTA
kr_3 = np.linspace(POLE_MAX, KR_MAX, N_TAIL)

dkr_1 = kr_1[1] - kr_1[0]
dkr_2 = kr_2[1] - kr_2[0]
dkr_3 = kr_3[1] - kr_3[0]

TAPER_FRACTION = 0.15
TAPER_LENGTH = int(N_TAIL * TAPER_FRACTION)
taper = np.ones(N_TAIL)
taper[-TAPER_LENGTH:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, TAPER_LENGTH)))

w_kr_3 = dkr_3 * taper
##################################################
########## NUMERICAL INTEGRATION SCRIPT ##########
##################################################