# Federated Learning with Differential Privacy - Comparative Study

## Overview
- **Summary**: A comprehensive comparative study of three federated learning algorithms: FedProx, DP-FedProx, and DP-Native-FedProx, focusing on their performance, privacy guarantees, and computational efficiency under identical experimental conditions.
- **Purpose**: To evaluate and compare the effectiveness of different approaches to incorporating differential privacy into federated learning, specifically in the context of FedProx.
- **Target Users**: Researchers and practitioners in federated learning and differential privacy, looking to understand the trade-offs between privacy preservation and model performance.

## Goals
- Conduct a fair, side-by-side comparison of FedProx, DP-FedProx, and DP-Native-FedProx algorithms
- Evaluate model performance (accuracy) and computational efficiency across all three algorithms
- Verify differential privacy compliance for DP-FedProx and DP-Native-FedProx
- Identify optimal hyperparameters for each algorithm through automated parameter tuning
- Provide a clear, reproducible experimental setup and results

## Non-Goals (Out of Scope)
- Modifying the core algorithmic implementations
- Testing with datasets other than MNIST
- Exploring additional privacy mechanisms beyond Gaussian noise
- Conducting experiments on non-CUDA hardware
- Implementing new federated learning algorithms

## Background & Context
- Federated learning enables model training across distributed devices without sharing raw data
- Differential privacy provides mathematical guarantees for privacy preservation
- FedProx introduces a proximal term to improve convergence in non-IID settings
- DP-FedProx adds differential privacy to FedProx through a wrapper approach
- DP-Native-FedProx integrates differential privacy directly into the local gradient computation

## Functional Requirements
- **FR-1**: Implement a unified experimental framework for all three algorithms
- **FR-2**: Support automated hyperparameter tuning for optimal performance
- **FR-3**: Provide real-time monitoring of training progress and performance metrics
- **FR-4**: Generate differential privacy compliance audit logs
- **FR-5**: Ensure fair comparison by using identical base parameters across all algorithms

## Non-Functional Requirements
- **NFR-1**: All experiments must run on CUDA-enabled hardware for optimal performance
- **NFR-2**: Training time per algorithm should be reasonable (under 1 hour per full run)
- **NFR-3**: Code should be well-documented and easily reproducible
- **NFR-4**: Differential privacy implementation must correctly apply noise and gradient clipping
- **NFR-5**: Parameter tuning should efficiently explore the search space

## Constraints
- **Technical**: PyTorch framework, CUDA-enabled hardware
- **Business**: No budget constraints, but computational efficiency is a evaluation metric
- **Dependencies**: MNIST dataset, PyTorch, NumPy

## Assumptions
- The provided implementations of FedProx, DP-FedProx, and DP-Native-FedProx are correct
- All experiments will be conducted on the same hardware to ensure fair comparison
- The MNIST dataset is already properly preprocessed and available

## Acceptance Criteria

### AC-1: Experiment Setup
- **Given**: A properly configured environment with PyTorch and CUDA
- **When**: Running the comparative experiment script
- **Then**: All three algorithms (FedProx, DP-FedProx, DP-Native-FedProx) should execute with identical base parameters
- **Verification**: `programmatic`

### AC-2: Hyperparameter Tuning
- **Given**: A defined search space for hyperparameters
- **When**: Running the automated tuning process
- **Then**: The system should identify optimal parameters for each algorithm based on accuracy and computational efficiency
- **Verification**: `programmatic`

### AC-3: Performance Comparison
- **Given**: Trained models for all three algorithms
- **When**: Evaluating on the MNIST test set
- **Then**: The system should generate a comprehensive comparison of accuracy, training time, and privacy budget consumption
- **Verification**: `programmatic`

### AC-4: Differential Privacy Compliance
- **Given**: Trained models for DP-FedProx and DP-Native-FedProx
- **When**: Analyzing the privacy audit logs
- **Then**: Both models should demonstrate (ε, δ)-differential privacy compliance with ε ∈ [0.5, 1]
- **Verification**: `programmatic`

### AC-5: Convergence Analysis
- **Given**: Training runs for all three algorithms
- **When**: Analyzing the training curves
- **Then**: All algorithms should show consistent accuracy improvement over training rounds
- **Verification**: `human-judgment`

## Open Questions
- [ ] What is the expected baseline accuracy for FedProx on MNIST with the given parameters?
- [ ] How much accuracy degradation is acceptable for privacy preservation?
- [ ] Are there any specific edge cases to consider when evaluating differential privacy compliance?
- [ ] What is the optimal balance between privacy budget and model performance?
