# Explainable Machine Learning for Band Gap Prediction (XML_bandgap)
(Last update: April 07, 2026)

- Project researchers at Toyota Central R&D Labs., Inc.: Joohwi Lee and Kaito Miyamoto

This repository provides a framework for feature selection/elimination based on feature importance:

- Support Vector Regression (SVR)
- Pair-correlation-based feature elimination
- Model-agonistic explainable machine learning analysis (XML): PFI, SHAP (ref: Lundberg and Lee, Adv. Neural Inf. Process. Syst. 30, 4768--4777 (2017))
- Out-of-domain (OOD) dataset

## Installation

Linux is recommended (macOS may also work).

The results in the paper were obtained using:
- Ubuntu 22.04
- Python 3.11

Last tested: April 07, 2026

Install required packages:

```bash
pip install numpy scipy scikit-learn shap matplotlib statsmodels seaborn
```

Download the in-domain (train/test) dataset from:
https://github.com/JoohwiLEE/GWgap_predictor_data
(ref: Lee et al., Phys. Rev. B, 93, 115104 (2016))

## Environment

| Package        | Version | Version |
|----------------|---------|---------|
| python         | 3.11.0  | 3.11.13 |
| pandas         | 3.0.2   | 2.3.0   |
| numpy          | 2.4.4   | 2.1.2   |
| scikit-learn   | 1.8.0   | 1.7.0   |
| scipy          | 1.17.1  | 1.15.3  |
| matplotlib     | 3.10.8  | 3.10.3  |
| seaborn        | 0.13.2  | 0.13.2  |
| statsmodels    | 0.14.6  | 0.14.6  |
| shap           | 0.51.0  | 0.48.0  |

Consistent results were obtained across two different environments.

## Code Flow

### 1. SVR with 18 features
Train pristine model with 18 features.
```bash
python svr-18feature.py | tee log-svr-18feature
```
Check 18_performance.csv and log.

### 2. Pair correlation analysis
Compute pairwise correlations.
```bash
python pair_correlation.py > log-pair-correlation
```
Check figure and log.

### 3. Pair-based feature elimination
Remove redundant features based on correlation and prediction error change.
```bash
python svr-pair-elimination.py | tee log-svr-pair-elimination
```
Check feature_elimination_history.csv and log.

### 4. SVR with 11 features
Train reduced model.
```bash
python svr-11feature.py | tee log-svr-11feature
```
Check 11_performance.csv and log.

### 5. PFI analysis
Compute PFI.
```bash
python svr-pfi-11feature.py | tee log-svr-pfi
```
Draw PFI graph.
```bash
python draw-svr-11feature-pfi-order.py
```

### 6. SHAP analysis
Compute SHAP values.
```bash
python svr-shap-11feature.py | tee log-svr-shap
```
Draw SHAP graph.
```bash
python draw-svr-11feature-shap-order.py
```
Check PFI/SHAP importance order.
```bash
python summary-svr-shap-order.py |tee log-pfi-shap
```

### 7. Feature count vs accuracy
Insert sorted feature indices (e.g., [1, 17, 10, 5, 14, 6, 9, 4, 12, 15, 16]). <br>
Train models with increasing feature count.
```bash
python svr-oneshot-2-to-11feature.py | tee log-svr-2-to-11
```
Check n_performance.csv (n = 2 to 11) and log. <br>
Plot results.
```bash
python draw-indomain-error.py > log-indomain
```
```bash
python draw-OOD-error.py > log-OOD
```
Check figures and logs.

## Citation
```
Joohwi Lee and Kaito Miyamoto, arXiv 2503.04492 (2025); submitted to Scientific Reports.
```
When accepted, the reference will be changed into an accepted journal.

## NOTICE

Copyright (C) 2026 TOYOTA CENTRAL R&D LABS., INC. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (collectively, the "Software"), the rights to use, copy, modify and/or merge the Software, for non-commercial research purposes, specifically limited to educational use, verification of research results, publication in academic papers, and presentations at academic conferences, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

Except as expressly stated herein, no rights or licenses from any copyright holder are granted, whether expressly, by implication, estoppel or otherwise.

The name and trademarks of copyright holder(s) may NOT be used in advertising or publicity pertaining to the Software or portions thereof including modifications or derivatives, without specific written prior permission. Title to copyright in the Software will at all times remain with the copyright holders.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

