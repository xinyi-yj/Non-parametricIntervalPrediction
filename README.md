# Non-parametricIntervalPrediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:  
**"A Deep Learning-based Model for Interval Prediction of Real-time Clearing Price in Electricity Market"** *(Currently under review / Accepted by [Journal Name])*.

## 💡 Highlights
* Time-shift feature construction and greedy selection optimize price dimensions.
* VMD and CEEMDAN effectively resolve extreme non-stationarity in clearing prices.
* Similarity-based non-parametric intervals dynamically quantify market uncertainty.
* Non-parametric interval breaks coverage-sharpness trade-off in quantile regression.

## 📝 Overview
Real-time electricity clearing prices are characterized by extreme volatility, non-stationarity, and complex multidimensional dependencies. While deep learning point forecasting models perform well, they fail to quantify the underlying uncertainty. 

This project provides a robust framework that:
1. **Optimizes Features**: Uses time-shift construction and greedy selection.
2. **Decomposes Signals**: Leverages VMD/CEEMDAN to handle extreme price non-stationarity.
3. **Generates Intervals**: Proposes a novel **non-parametric similarity-based interval generator** that breaks the severe "coverage-sharpness trade-off" often suffered by end-to-end parametric models like Quantile Regression (LSTM-QR and the SOTA TCN-QR).

## 📂 Repository Structure

```text
├── data/                    # Placeholder for dataset (See Data Availability)
├── models/
│   ├── lstm_model.py        # Base LSTM for point prediction
│   ├── tcn_qr.py            # SOTA Temporal Convolutional Network with Quantile Regression (TCN-QR)
│   └── lstm_qr.py           # LSTM with Quantile Regression
├── utils/
│   ├── feature_selection.py # Greedy-based feature selection & Time-shift construction
│   ├── decomposition.py     # CEEMDAN and VMD implementations
│   └── metrics.py           # Evaluation metrics (PICP, MIW, MPICD, Pinball Loss)
├── interval_generator/
│   └── similarity_match.py  # Non-parametric interval construction module
├── main_point_pred.py       # Training and evaluation for point prediction
├── main_interval_pred.py    # Training and evaluation for interval prediction & baselines
└── requirements.txt         # Dependencies
