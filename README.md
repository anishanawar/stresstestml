## StressTestML — Robustness & Safety Evaluation for ML Models

### Overview

StressTestML is a safety-aware machine learning evaluation framework designed to analyze how models behave under:

* Small input perturbations
* Distribution shifts
* High-confidence failure scenarios
* Calibration mismatch
* Selective prediction (abstention)

This project demonstrates how models can fail silently and how to engineer safeguards around model confidence in higher-stakes decision settings.

## Problem Motivation

In high-impact systems (finance, fraud detection, healthcare triage, moderation pipelines), raw accuracy is not enough.

Key risks include:

* High-confidence incorrect predictions
* Performance degradation under small input changes
* Poor calibration (confidence ≠ correctness)
* Unsafe predictions on uncertain inputs

StressTestML addresses these risks through structured robustness evaluation and safety gating.

## Features Implemented

### 1. Baseline Model

* Logistic Regression classifier
* Standardized preprocessing pipeline
* ROC-AUC evaluation

### 2. Robustness Testing

Gaussian feature perturbations simulate small realistic input changes.

Measures:

* AUC degradation
* High-confidence wrong predictions (≥ 0.9 probability)

### 3. High-Confidence Error Tracking

Tracks “silent failures” where the model is very confident but incorrect.

This is critical for safety-sensitive systems.

### 4. Calibration Analysis

Implements Expected Calibration Error (ECE) to measure:

> Does 90% confidence actually mean 90% accuracy?

Poor calibration increases operational risk.

### 5. Abstention (Reject Option)

Implements confidence gating:

If prediction probability is between **0.4–0.6 → abstain.**

Reports:

* Coverage (how often the model predicts)
* Accuracy on accepted predictions
* High-confidence wrong after gating

This simulates a safety review layer before final decisions.

## Dataset

UCI Default of Credit Card Clients dataset (.xls format)

Used as a proxy for financial risk modeling.
