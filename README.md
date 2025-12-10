
# Resampling-Based Random Fourier Features and Mixture-of-GAMs

## 1. Overview
This repository contains the MATLAB implementation of a resampling-based Random Fourier Feature (RFF) method and the corresponding Mixture-of-Generalized Additive Models (Mixture-of-GAMs) framework. The goal is to combine kernel-inspired feature learning with interpretable local models to improve predictive performance while retaining transparency.

The method is applied to two regression datasets:
- **California Housing**
- **NASA Airfoil Self-Noise**

The code implements:
- Adaptive resampling of Fourier frequencies  
- PCA extraction of learned spatial structure  
- Gaussian mixture clustering  
- Cluster-wise GAM fitting with smooth components  
- Mixture prediction using posterior responsibilities  

This repository focuses on the *core MATLAB implementation*. Baseline models and Python notebooks may be added later.

---

## 2. Workflow Summary
The end-to-end workflow consists of the following steps:

1. **Train RFF Model with Resampling**  
   The frequency distribution is iteratively updated using a resampling mechanism that concentrates mass along directions capturing dominant variation.

2. **Extract RFF Embedding and Apply PCA**  
   PCA is used to identify principal directions in the learned frequency distribution, providing interpretable low-dimensional structure.

3. **Fit a Gaussian Mixture Model (GMM)**  
   The PCA embedding is clustered using a soft GMM, producing cluster responsibilities for each data point.

4. **Train Local GAMs**  
   A separate generalized additive model is fitted in each cluster using univariate smooth functions.

5. **Predict with Mixture-of-GAMs**  
   Final predictions are formed by weighting cluster-specific GAM outputs by their posterior responsibilities.

If included, the workflow diagram appears below:

![Workflow](docs/figures/Workflow_diagram.png)

---

## 3. Results and Visualization
The repository includes scripts for generating summary visualizations such as:

- **RMSE comparisons** between RFF, global GAM, and Mixture-of-GAMs  
- **Learned RFF frequency distributions**  
- **PCA principal direction analysis**  
- **Partial dependence plots** for local GAM components  

Example figures (to be placed in `docs/figures/`):

<!--
```markdown
![RMSE Comparison](docs/figures/rmse_california.png)
![RFF Frequency Distribution](docs/figures/rff_distribution.png)
![PDP Components](docs/figures/pdp_components.png)
-->