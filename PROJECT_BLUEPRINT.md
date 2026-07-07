# Project: Federated Learning for DeFi/Crypto Hedging (FedHedge)
**Status:** Phase 1 - Data Pipeline & Environment Setup
**Goal:** Train a global hedging model across fragmented exchanges (CEX/DEX) without sharing raw data, solving the Non-IID challenge.

## Tech Stack
* **Machine Learning:** PyTorch
* **Federated Learning:** Flower (flwr.dev)
* **Data/Finance:** CCXT, Pandas, NumPy
* **Environment:** Python 3.10+, Docker (Later)

## Architecture Overview
1.  `data/`: Stores raw and processed data (simulating isolated exchange data).
2.  `clients/`: Local models representing isolated exchanges.
3.  `server/`: The global federated aggregator.
4.  `models/`: PyTorch neural network definitions.

## Project Roadmap (Checklist)
- [x] Phase 0: Conceptualization & Blueprint creation.
- [x] Phase 1: Environment Setup & Data Pipeline (Fetching CCXT data).
- [x] Phase 2: Feature Engineering & Non-IID simulation (Splitting data per client).
- [x] Phase 3: Local Model Development (Deep Hedging baseline in PyTorch).
- [x] Phase 4: Federated Learning Integration (Flower Server/Client setup).
- [x] Phase 5: Evaluation, Backtesting & Metrics (Hedging Error, cVaR).
- [/] Phase 6: Open Source Packaging (Docker, GitHub Actions).

## Current Context for AI
We are currently executing Phase 1: Setting up the directory structure and writing the initial script to fetch historical OHLCV data from multiple exchanges using CCXT to simulate isolated environments.