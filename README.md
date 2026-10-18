# Deep Learning Network Intrusion Detector (Autoencoder)

## Overview
This project implements an unsupervised Deep Learning model using PyTorch to detect network intrusions and anomalies. Unlike traditional signature-based detection, this model uses an **Autoencoder Neural Network** to learn the baseline of "normal" network traffic and flag deviations.

## Dataset
The model is trained on the **NSL-KDD** dataset, a benchmark dataset for network intrusion detection. 
- Training data consists **only** of normal network flows.
- Testing data contains a mix of normal flows and various attacks (DDoS, Probe, R2L, U2R).

## Neural Network Architecture
The Autoencoder is built using PyTorch (`torch.nn`). It consists of two parts:
1. **Encoder:** Compresses the 41 input features down to a dense bottleneck of 4 neurons using Linear layers and ReLU activation.
2. **Decoder:** Reconstructs the original 41 features from the 4-neuron bottleneck.

The network is trained using Mean Squared Error (MSE) loss and the Adam optimizer. Because it only sees normal traffic during training, it learns to reconstruct normal flows with very low error.

## Anomaly Detection Logic
When testing, the model calculates the **Reconstruction Error** (MSE) for each network flow. 
- If the error is low, the flow is Normal (the model recognized it).
- If the error is high, the flow is an Attack (the model has never seen this pattern before).

### Threshold Tuning
The most critical part of this project was threshold tuning. An initial statistical threshold (95th percentile) resulted in a 72% accuracy but only 9% attack recall. By manually testing thresholds and analyzing the precision-recall trade-off, the threshold was set to `0.1`, resulting in:
- **Attack Recall:** 93%
- **Attack F1-Score:** 0.82
- **Overall Accuracy:** 79%

This demonstrates the practical reality of deploying AI in cybersecurity: minimizing false negatives (missed attacks) is prioritized over minimizing false positives.

## How to Run
1. Download the NSL-KDD dataset (`KDDTrain+.txt` and `KDDTest+.txt`).
2. Install dependencies: `pip install torch pandas scikit-learn joblib`
3. Train the model: `python train_autoencoder.py`
4. Evaluate thresholds: `python evaluate_anomaly.py`
