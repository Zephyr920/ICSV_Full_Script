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
DISK_RADIUS = 0.05                                    # Radius of disk, a (to be chosen by user)
ETA = 0.00                                            # Loss Factor (to be chosen by user): 0.00 - lossless environment
                                                      #                                     0.01 - small damping for soil
                                                      #                                     0.03 - large damping for soil
kc = OMEGA / CP                                       # Compressional wavenumber - To add damping CP -> (CP * (1 - 1j * ETA))
ks = WAVENUMBER_RATIO * kc                            # Shear wavenumber

MU = RHO * (CP / WAVENUMBER_RATIO)**2                 # Lamé Parameter - mu
C = RHO * CP**2                                       # Lamé Parameters - C = lambda + 2 * mu
LAMBDA = C - 2 * MU                                   # Lamé Parameter - lambda

CR = (OMEGA / ks) * ((0.87 + 1.12 * NU) / (1 + NU))   # Rayleigh wave speed
LAMBDA_R = CR / FREQUENCY                             # Rayleigh wavelength - Note you can use 2 * pi * (1/POLE)
kR = OMEGA / CR                                       # Rayleigh wave number

# Physical domain parameters - these serve us well for low frequencies
R_MIN = 1e-10
R_MAX = 1000
R_POINTS = 100
Z_SCALE = 3

# For high frequencies (2000Hz) our wavelengths become tiny so the previous Physical domain parameters are insufficient.
NUMBER_OF_WAVELENGTHS = 10
POINTS_PER_WAVELENGTH = 20
#R_MAX = (NUMBER_OF_WAVELENGTHS * LAMBDA_R).real                  # Whilst you can use these for low frequencies
#R_POINTS = (NUMBER_OF_WAVELENGTHS * POINTS_PER_WAVELENGTH).real  # with a HUGE physical distance covered, these changes
                                                                  # become critical for higher frequencies.

# Arrays
r = np.linspace(R_MIN, R_MAX, R_POINTS)
z = np.linspace(Z_SCALE * R_MIN, Z_SCALE * R_MAX, Z_SCALE * R_POINTS) # NOTE: Z_SCALE ensures that we integrate beyond
                                                                      #       the semi-sphere to fully encapsulate the
                                                                      #       depth reaching infinity for the full field
                                                                      #       Additionally, 1e-10 * 3 is approximately 0
                                                                      #       from a numerical perspective.
                                                                      #       We require 1e-10 for r because in the div(u)
                                                                      #       We will return a (u / r) term.
dz = z[1] - z[0]

# Dynamically calculate the Rayleigh pole each time we change a parameter.
def find_poles():
    cubic_coeff = [(16 * MU**2 * kc**2 - 16 * MU * C * kc**2), (4 * MU**2 * ks**4 + 16 * MU * C * kc**2 * ks**2 + 4 * C**2 * kc**4 - 16 * MU**2 * kc**2 * ks**2),
               (-4 * MU * C * kc**2 * ks**4 - 4 * C**2 * kc**4 * ks**2), (C**2 * kc**4 * ks**4)] # This is Eq. 3.2
    X = np.roots(cubic_coeff)             # poles^2
    condition = X[np.imag >= 0]
    zeta = sqrt_safe(condition)
    rayleigh_pole = zeta[zeta**2 > ks**2] # Isolate just the Rayleigh pole

    return rayleigh_pole

POLE = find_poles().astype(complex)   # returns an array
POLE = POLE[0]                        # return just a complex number

##############################################################
############### INTEGRATION FUNCTION'S KERNELS ###############
##############################################################
def curly_phi(k_r):
    numerator = P0 * DISK_RADIUS * jv(1, k_r * DISK_RADIUS)                                                       # Might need (2 * np.pi) and/or delta(w - w0)
    denominator_term_1 = (ks**2 - 2 * k_r**2) * ((LAMBDA + 2 * MU) * kc**2 - 2 * MU * k_r**2)
    denominator_term_2 = - 4 * MU * k_r**2 * sqrt_safe(k_r**2 - ks**2) * sqrt_safe(k_r**2 - kc**2)
    return numerator / (denominator_term_1 + denominator_term_2)

def phi_kernel(k_r):
    numerator = (ks**2 - 2 * k_r**2) * curly_phi(k_r)
    denominator = k_r
    return numerator / denominator

def Phi_kernel(k_r):
    return 2 * sqrt_safe(k_r**2 - kc**2) * curly_phi(k_r)

def exponential_term(sqrt_argument):
    return np.exp(-z[np.newaxis,:] * sqrt_safe(sqrt_argument[:, np.newaxis]))

def bessel_term(order, k_r):
    return jv(order, k_r[:, np.newaxis] * r[np.newaxis, :])

# Note our fields are coupled in the scenario presented here so the Rayleigh contribution will be embedded
#                                                                            in our potential expressions.
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

# The integration is split in this way to best align with the paper's methodology and to allow easier parallelisation.
kr_1 = np.linspace(start=0, stop=POLE_MIN, num=R_POINTS) + 1j * DELTA            # Pre-pole region
kr_2 = np.linspace(start=POLE_MIN, stop=POLE_MAX, num=R_POINTS) + 1j * DELTA     # Pole region
kr_3 = np.linspace(POLE_MAX, KR_MAX, N_TAIL)                                     # Cosine Tapering Region

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
# -------------- PRE-COMPUTATIONS --------------
# Please note _1, _2, _3 refer to the region the array is tied to: Pre-pole, pole, taper.
# This is the precomputation stage

# Horizontal Component
scalar_potential_u_r_kernel_1 = comp_u_r_kernel(kr_1)
scalar_potential_u_r_kernel_2 = comp_u_r_kernel(kr_2[1:])  # Due to the end-point of kr_1 being the start of kr_2
scalar_potential_u_r_kernel_3 = comp_u_r_kernel(kr_3)

vector_potential_u_r_kernel_1 = shear_u_r_kernel(kr_1)
vector_potential_u_r_kernel_2 = shear_u_r_kernel(kr_2[1:])
vector_potential_u_r_kernel_3 = shear_u_r_kernel(kr_3)

# Vertical Component
scalar_potential_u_z_kernel_1 = comp_u_z_kernel(kr_1)
scalar_potential_u_z_kernel_2 = comp_u_z_kernel(kr_2[1:])
scalar_potential_u_z_kernel_3 = comp_u_z_kernel(kr_3)

vector_potential_u_z_kernel_1 = shear_u_z_kernel(kr_1)
vector_potential_u_z_kernel_2 = shear_u_z_kernel(kr_2[1:])
vector_potential_u_z_kernel_3 = shear_u_z_kernel(kr_3)

# Radial Component from the Inverse Hankel Transforms of Eqs. (2.8, 2.9)
J1_1 = bessel_term(1, kr_1)
J1_2 = bessel_term(1, kr_2[1:])
J1_3 = bessel_term(1, kr_3)

J0_1 = bessel_term(0, kr_1)
J0_2 = bessel_term(0, kr_2[1:])
J0_3 = bessel_term(0, kr_3)

# Exponential Components from Eqs. (2.8, 2.9) in form of Simple Harmonic Oscillator as we're solving Eqs. (2.2, 2.3) in z
scalar_potential_exp_1 = exponential_term(kr_1**2 - kc**2)
scalar_potential_exp_2 = exponential_term(kr_2[1:]**2 - kc**2)
scalar_potential_exp_3 = exponential_term(kr_3**2 - kc**2)

vector_potential_exp_1 = exponential_term(kr_1**2 - ks**2)
vector_potential_exp_2 = exponential_term(kr_2[1:]**2 - ks**2)
vector_potential_exp_3 = exponential_term(kr_3**2 - ks**2)

# ----- Numerical Integration (einsum) -----
# Inverse Hankel Transforming the three regions: 1 - pre-pole, 2 - pole, 3 - tail
scalar_potential_u_r_1 = dkr_1 * np.einsum("ij,ik -> jk", scalar_potential_u_r_kernel_1 * J1_1, scalar_potential_exp_1)
scalar_potential_u_r_2 = dkr_2 * np.einsum("ij,ik -> jk", scalar_potential_u_r_kernel_2 * J1_2, scalar_potential_exp_2)
scalar_potential_u_r_3 = np.einsum("i,ij,ik -> jk", w_kr_3, scalar_potential_u_r_kernel_3 * J1_3, scalar_potential_exp_3)

vector_potential_u_r_1 = dkr_1 * np.einsum("ij,ik -> jk", vector_potential_u_r_kernel_1 * J1_1, vector_potential_exp_1)
vector_potential_u_r_2 = dkr_2 * np.einsum("ij,ik -> jk", vector_potential_u_r_kernel_2 * J1_2, vector_potential_exp_2)
vector_potential_u_r_3 = np.einsum("i,ij,ik -> jk", w_kr_3, vector_potential_u_r_kernel_3 * J1_3, vector_potential_exp_3)

scalar_potential_u_z_1 = dkr_1 * np.einsum("ij,ik -> jk", scalar_potential_u_z_kernel_1 * J0_1, scalar_potential_exp_1)
scalar_potential_u_z_2 = dkr_2 * np.einsum("ij,ik -> jk", scalar_potential_u_z_kernel_2 * J0_2, scalar_potential_exp_2)
scalar_potential_u_z_3 = np.einsum("i,ij,ik -> jk", w_kr_3, scalar_potential_u_z_kernel_3 * J0_3, scalar_potential_exp_3)

vector_potential_u_z_1 = dkr_1 * np.einsum("ij,ik -> jk", vector_potential_u_z_kernel_1 * J0_1, vector_potential_exp_1)
vector_potential_u_z_2 = dkr_2 * np.einsum("ij,ik -> jk", vector_potential_u_z_kernel_2 * J0_2, vector_potential_exp_2)
vector_potential_u_z_3 = np.einsum("i,ij,ik -> jk", w_kr_3, vector_potential_u_z_kernel_3 * J0_3, vector_potential_exp_3)

# Adding each integration regions contribution will give us the compressional/shear AND Rayleigh contribution
# Associated with each potential
compressional_and_rayleigh_u_r = scalar_potential_u_r_1 + scalar_potential_u_r_2 + scalar_potential_u_r_3
compressional_and_rayleigh_u_z = scalar_potential_u_z_1 + scalar_potential_u_z_2 + scalar_potential_u_z_3

shear_and_rayleigh_u_r = vector_potential_u_r_1 + vector_potential_u_r_2 + vector_potential_u_r_3
shear_and_rayleigh_u_z = vector_potential_u_z_1 + vector_potential_u_z_2 + vector_potential_u_z_3

# Gives us the full field horizontal and vertical displacements
u_r = compressional_and_rayleigh_u_r + shear_and_rayleigh_u_r
u_z = compressional_and_rayleigh_u_z + shear_and_rayleigh_u_z

# Depending on array sizing and memory available one can delete all vector_potential_ and scalar_potential_ arrays up until this stage
del scalar_potential_u_r_kernel_1, scalar_potential_u_r_kernel_2, scalar_potential_u_r_kernel_3, scalar_potential_u_z_kernel_1, scalar_potential_u_z_kernel_2, scalar_potential_u_z_kernel_3
del vector_potential_u_r_kernel_1, vector_potential_u_r_kernel_2, vector_potential_u_r_kernel_3, vector_potential_u_z_kernel_1, vector_potential_u_z_kernel_2, vector_potential_u_z_kernel_3
del J0_1, J0_2, J0_3, J1_1, J1_2, J1_3
del scalar_potential_exp_1, scalar_potential_exp_2, scalar_potential_exp_3, vector_potential_exp_1, vector_potential_exp_2, vector_potential_exp_3

# If we continue with the compressional_and_rayleigh or shear_and_rayleigh by applying the power radiated expressions
# From Eqs. (2.14), or using spherical coordinates using (2.15)