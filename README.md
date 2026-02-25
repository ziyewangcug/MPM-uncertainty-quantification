# MPM-uncertainty-quantification
Uncertainty quantification of deep learning model for mineral prospectivity mapping

This repository provides a deep learning framework for Mineral Prospectivity Mapping (MPM) that integrates both:

Data uncertainty (via stochastic simulation of evidence layers)

Model uncertainty (via Bayesian CNN with Monte Carlo Dropout)

# Overview
Mineral prospectivity mapping often suffers from:

Uncertainty in geological and geochemical evidence layers

Limited labeled deposit samples

Overconfident deterministic deep learning predictions

# This project addresses these challenges by:
Generating multiple simulated evidence layers

Training a Bayesian CNN on each simulation

Performing Monte Carlo Dropout inference

Directly computing predictive variance and entropy

# The final outputs include:
Mean prediction map

Variance (uncertainty) map

Entropy map

Full prediction ensemble

# Requirements
Python ≥ 3.8

TensorFlow ≥ 2.x

NumPy

SciPy

GDAL

scikit-learn

matplotlib

# Input Data
Deposit Label File

deposit.tif

Binary raster

1 = deposit location

0 = non-deposit

Simulated Evidence Layers

evidence layers.mat

MATLAB structure:

combined_layers[0...n]

Each element shape:

(H, W, m)

Where:

H = height

W = width

m = number of evidence layers
