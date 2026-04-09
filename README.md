# FRE-GY 7773 - Machine Learning for Financial Engineering

[Course Syllabus](https://engineering.nyu.edu/sites/default/files/2025-11/FRE-GY7773syllabus-fb.pdf)

This repository contains the course materials for FRE-GY 7773 - Machine Learning for Financial Engineering taught at NYU Tandon School of Engineering.

## Notebooks

Every notebook is accompanied by a link to a corresponding Google Colab session for interactive execution.

### Lecture 1 (Introduction to Python, Jupyter, and ML Tools)

- [01_linalg.ipynb](./01_linalg.ipynb) - Linear Algebra examples.
- [01_probability_statistics.ipynb](./01_probability_statistics.ipynb) - Probability/Stats examples.
- [01_productivity.md](./01_productivity.md) - Productivity Review.

- Notebooks taken from the GitHub Repo [handson-mlp](https://github.com/ageron/handson-mlp) by Aurélien Géron.
  - [01_tools_matplotlib.ipynb](./01_tools_matplotlib.ipynb) - Introduction to Matplotlib for data visualization.
  - [01_tools_numpy.ipynb](./01_tools_numpy.ipynb) - Introduction to NumPy for numerical computing.
  - [01_tools_pandas.ipynb](./01_tools_pandas.ipynb) - Introduction to Pandas for data manipulation.

### Lecture 2 (MLE / Data in Finance)

- [02_mle.ipynb](./02_mle.ipynb) - Maximum Likelihood Estimation examples.
- [02_mle_solution.ipynb](./02_mle_solution.ipynb) - Maximum Likelihood Estimation examples - Solutions.
- [02_data_finance.ipynb](./02_data_finance.ipynb) - Data in Finance examples.
- [02_homework.ipynb](./02_homework.ipynb) - Homework 1.

## Lecture 3 (Linear Regression)

- [03_linear_regression.ipynb](./03_linear_regression.ipynb) - Linear Regression examples.
- [03_linear_regression_solution.ipynb](./03_linear_regression_solution.ipynb) - Linear Regression examples - Solutions.

## Lecture 4 (Ridge/Lasso)

- [04_ridge_lasso.ipynb](./04_ridge_lasso.ipynb) - Ridge and Lasso Regression examples. Train/validation/test sets, cross-validation, pipeline.
- [04_ridge_lasso_solution.ipynb](./04_ridge_lasso_solution.ipynb) - Ridge and Lasso Regression examples - Solutions.

## Lecture 5 (Logistic Regression - Classification)

- [05_classification.ipynb](./05_classification.ipynb) - MNIST classification example.
- [05_classification_solution.ipynb](./05_classification_solution.ipynb) - MNIST classification example - Solutions.
- [05_homework.ipynb](./05_homework.ipynb) - Homework 2.

## Lecture 6 (Optimization)

- [06_optimization.ipynb](./06_optimization.ipynb) - Optimization examples for Linear Regression.
- [06_optimization_solution.ipynb](./06_optimization_solution.ipynb) - Optimization examples for Linear Regression - Solutions.
- [06_optim_logreg.py](./06_optim_logreg.py) - Optimization examples for Logistic Regression.
- [06_optim_logreg_solution.py](./06_optim_logreg_solution.py) - Optimization examples for Logistic Regression -Solutions.

## Lecture 7 (PCA)

- [07_gaussian_pca.ipynb](./07_gaussian_pca.ipynb) - PCA example for Gaussian data.
- [07_gaussian_pca_solution.ipynb](./07_gaussian_pca_solution.ipynb) - PCA example for Gaussian data - Solutions.
- [07_yield_curve_pca.ipynb](./07_yield_curve_pca.ipynb) - PCA example for yield curves.
- [07_yield_curve_pca_solution.ipynb](./07_yield_curve_pca_solution.ipynb) - PCA example for yield curves - Solutions.
- [07_homework.ipynb](./07_homework.ipynb) - Homework 3.

## Lecture 8 (K-Means and Gaussian Mixture Models)

- [08_k_means.ipynb](./08_k_means.ipynb) - K-Means clustering example.
- [08_k_means_solution.ipynb](./08_k_means_solution.ipynb) - K-Means clustering example - Solutions.
- [08_gaussian_mixture_models.ipynb](./08_gaussian_mixture_models.ipynb) - Gaussian Mixture Models example.
- [08_gaussian_mixture_models_solution.ipynb](./08_gaussian_mixture_models_solution.ipynb) - Gaussian Mixture Models example - Solutions.
- [08_pca_k_means_returns.ipynb](./08_pca_k_means_returns.ipynb) - PCA + K-Means example for stock returns.
- [08_pca_k_means_returns_solution.ipynb](./08_pca_k_means_returns_solution.ipynb) - PCA + K-Means example for stock returns - Solutions.

## Lecture 9 (Neural Networks and PyTorch)

## Installation

1. Clone this repository and navigate to the project directory:

   ```bash
   git clone https://github.com/fbourgey/fre-gy-7773-mlfe.git
   cd fre-gy-7773-mlfe
   ```

2. Synchronize with [uv](https://docs.astral.sh/uv/)

   ```bash
   uv sync
   ```
