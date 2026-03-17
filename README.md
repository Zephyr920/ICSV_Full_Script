# ICSV 32 - Using numerical modelling to assess the potential to measure Rayleigh wave intensity.
##### Authors
1. Ioan-Alexandru Stancioiu - I.Stancioiu@liverpool.ac.uk
2. Carl Hopkins - carl.hopkins@liverpool.ac.uk

This repository provides a reproducible Python implementation of the methods described in:

"Using Numerical Modeling to Assess the Potential to Measure Rayleigh Wave Intensity"

The aim is to reproduce the displacement fields, intensity, power radiated calculations, and figures presented in the paper for each of the three wave types, in cylindrical and spherical coordinates.
The results are stored as .csv files so that the model can be manipulated in any way seen fit. All graphics produced here are included in the accompanying conference paper, but not all figures in the conference paper will be replicated here (Figure 1.).

## Overview
The script performs the following steps:
1. Computes displacement fields using:
   - Inverse Hankel Transform (IHT)
   - Cauchy's Residue Theorem (CRT)
   - Far-Field Approximation (FFD)
2. Evaluates intensity and power radiated with a cylindrical and spherical coordinate system
3. Stores immediate results in .csv format in "data" folder
4. Generates plots from computed data and stores in "graphs" folder

Note the time taken for each step of the process is also given as the code is run.

## Repository Structure
| File             |                                 Description                                  |
|------------------|:----------------------------------------------------------------------------:|
| main.py          |                 Main Script the ICSV paper is based off of.                  |
| data/            | Folder containing all displacement field, intensity and power radiated data. |
| graphs/          |                      Folder containing all plots saved                       |
| requirements.txt |                             Library dependencies                             |
| README.md        |                             Overview of program                              |

## Equation Tracker