# Gaussian Discriminant Analysis Classifier

## Overview

This project implements a Gaussian Discriminant Analysis (GDA) classifier, supporting both **univariate** and **multivariate** Gaussian models. It provides visual insights into discriminant functions, decision boundaries, and the effects of using pooled covariance matrices and modified class priors.

## Features

- Univariate and Multivariate GDA
- Support for multiple classes
- Class-specific and pooled covariance matrices
- Adjustable prior probabilities
- Visualization of discriminant functions and decision boundaries

## Univariate GDA

### Description

Two Gaussian distributions are modeled:
- **Resting**: Mean = 60, Std Dev = 5
- **Stressed**: Mean = 100, Std Dev = 5

Discriminant functions are computed for the range \( x = 40 \) to \( x = 120 \), and the decision boundary is identified where the discriminant functions intersect.

### Insights

- **Why they cross at 80**: The means are symmetric and the variances are equal, leading to the crossing point at the midpoint.
- **Parabolic shape**: The discriminant function includes a quadratic term from the Gaussian log-likelihood, resulting in a parabolic curve.

## Multivariate GDA

### Description

Multivariate distributions are modeled for two classes:
- **Resting**: Mean = [60, 10], Covariance = [[20, 100], [100, 20]]
- **Stressed**: Mean = [100, 80], Covariance = [[50, 20], [20, 50]]

The discriminant functions are computed for 2D input vectors from [20, 20] to [120, 120], and decision boundaries are visualized.

## General Discriminant Classifier

This module integrates the univariate and multivariate implementations into a reusable classifier for multiple classes.

### Key Capabilities

- Class setup without re-fitting
- Pooled covariance matrix computation
- Prior probability adjustment

### Testing

- Decision regions are visualized for multivariate GDA
- Using pooled covariance results in **linear** boundaries
- Changing priors shifts the boundaries **toward the more probable class**

## Dependencies

- Python 3.x
- NumPy
- Matplotlib

Install with:

```bash
pip install numpy matplotlib
```

## Repository Structure

```
.
├── discriminant.py                                              # descriminant model implementations
├── classifier.py                                                # classifier implementation
├── Univariate_and_Multivariate_Discriminants.ipynb              # Entire Notebook Implementation
├── README.md                                                    # Project documentation
```
