# FEDHEDGE: MASTER SYSTEM SPECIFICATION & CONTEXT BACKUP

## 1. PROJECT OVERVIEW & PURPOSE

**Project Name:** FedHedge
**Core Definition:** An Institutional-Grade, Federated Reinforcement Learning (FL + RL) quantitative trading and dynamic hedging system.
**Primary Objective:** To generate alpha and manage portfolio variance in the medium-term timeframe (1-2 weeks) across multiple cryptocurrency exchanges (Binance, OKX, Kraken). The system prioritizes risk-adjusted returns, out-of-sample robustness, and extreme tail-risk mitigation over raw directional accuracy.
**Key Philosophy:** "Accuracy is irrelevant if risk management fails." The system acts as a decentralized risk manager, calculating dynamic hedge ratios to protect capital during regime shifts (bear markets) while minimizing hedging costs during bull runs.

---

## 2. SYSTEM INPUTS & OUTPUTS

### 2.1. System Inputs (State Space)

* **Raw Data (Historical & Live):** * OHLCV (Open, High, Low, Close, Volume) data anchored to a 1H or 4H timeframe.
* Derivative Market Data: Funding Rates, Open Interest (OI).


* **Processed Features (The AI's Vision):**
* *Stationary Price Action:* Logarithmic returns, Fractional Differentiation series.
* *Regime Filters:* Zero-lag Kalman Filter outputs, Moving Averages (e.g., MA99 distance to price).
* *Volatility Metrics:* Rolling Standard Deviation, Historical Value at Risk (VaR).
* *Normalized Indicators:* Rolling Z-scores of volume and momentum to strictly prevent Look-ahead bias.


* **Federated Inputs:** Global Neural Network Weights broadcasted from the Central FL Server to local exchange clients.

### 2.2. System Outputs (Action Space)

* **Trading Actions:** * `Dynamic Hedge Ratio (h)`: A continuous value between $0.0$ and $1.0$ dictating the percentage of the spot portfolio to short via perpetual futures.
* `Position Sizing`: Capital allocation adjustments based on Kelly Criterion or volatility parity.


* **Federated Outputs:** Updated Local Neural Network Weights (gradients) sent back from local clients to the Central FL Server for aggregation.

---

## 3. ARCHITECTURE & DESIGN PATTERNS

### 3.1. High-Level Architecture

The system is strictly modularized to decouple data engineering, model training, and backtesting.

* **Data Alignment Module:** Handles asynchronous data streams from different exchanges. Uses a chosen Anchor Exchange (e.g., Binance) and applies Forward Fill (`ffill`) to prevent timestamp mismatch and `NaN` propagation.
* **Federated Server (Flower Framework):** Implements the **FedProx** aggregation algorithm. This is critical to prevent "Client Drift" caused by Non-IID data (different liquidity/volatility profiles across exchanges).
* **RL Trading Client:** Utilizes advanced algorithms (e.g., PPO or SAC) with **Prioritized Experience Replay** to ensure the AI does not suffer from "Catastrophic Forgetting" of historical market crashes.
* **Event-Driven Backtester:** A tick-by-tick or candle-by-candle simulation environment that strictly incorporates execution latency, maker/taker fees, volume-based slippage, and funding rate decay.

### 3.2. Software Design Patterns

To ensure infinite scalability and maintainability, FedHedge strictly implements the following Object-Oriented Design Patterns:

* **Strategy Pattern:** Used for Feature Engineering (`BaseFeatureEngineer`) and Reward Functions (`BaseRewardFunction`). Allows hot-swapping between a Sharpe-based reward or a Maximum Drawdown-based reward without altering the core RL agent.
* **Factory Pattern:** Used to instantiate RL Models (`ModelFactory.create("PPO")`) and Trading Environments. Ensures that adding a new algorithm (e.g., Transformer) requires zero changes to the main training loop.
* **Observer Pattern:** Implemented in the Live Trading module via WebSockets. The trading engine "listens" to the data stream and triggers the RL Agent's `predict_action()` method asynchronously.
* **Singleton Pattern:** Used for global configuration managers (`ConfigManager`) and unified system loggers to prevent memory leaks and I/O bottlenecks.

---

## 4. TRADING STRATEGY & RISK MANAGEMENT LOGIC

* **Timeframe Target:** Medium-term (Swing Trading / 1-2 weeks holding period) to negate local WiFi latency and minimize the impact of HFT (High-Frequency Trading) slippage and overtrading fees.
* **Market Regime Switching:** Uses lagging indicators (MA99) and zero-lag filters (Kalman) not for predictive entry, but as "Regime Filters." The RL agent learns to aggressively hedge when $Price < MA99$ (Bear Regime) and relax the hedge when $Price > MA99$ (Bull Regime).
* **Risk-Adjusted Reward Shaping:** The RL Agent is NEVER rewarded purely on PnL. The reward function is based on the **Sortino Ratio** (penalizing only downside volatility) minus transaction costs.
* **Hard Circuit Breakers:** A deterministic, non-AI logic layer that overrides the RL model. If daily drawdown exceeds a predefined threshold (e.g., -5%), the circuit breaker liquidates all positions and halts the system, protecting against Black Swan events where the AI might behave unpredictably.

---

## 5. STRICT CODING STANDARDS & FORMATTING

* **PEP 8 Compliance:** All Python code must strictly adhere to PEP 8 standards.
* **Type Hinting:** Mandatory across the entire codebase. Every function argument and return type must be explicitly typed (e.g., `def calculate_var(returns: pd.Series) -> float:`).
* **Docstrings:** Google-style docstrings are mandatory for all classes and functions, explicitly detailing `Args:`, `Returns:`, and `Raises:`.
* **Vectorization over Iteration:** No `for` loops for data processing. All feature engineering and risk metrics must utilize `numpy` matrix operations and `pandas` vectorized methods for performance.
* **Checkpointing:** Models must be saved securely as `.pth` files encompassing `model_state_dict`, `optimizer_state_dict`, `current_epoch`, and the `replay_buffer` to allow seamless Incremental Learning across different machines.

---

## 6. CRITICAL RISK MITIGATION (THE "PRE-MORTEM" DIRECTIVES)

To prevent institutional failure, the system enforces these strict rules:

1. **Anti-Look-Ahead Bias:** All computed features must be explicitly shifted (`.shift(1)`) before being passed to the RL State space. Normalization (e.g., Z-scores) must ONLY use rolling windows (`rolling(window=N).mean()`), never global means.
2. **Walk-Forward Analysis (WFA):** Traditional Train/Test splits are banned. Backtesting must utilize expanding or rolling windows (e.g., Train 2021 -> Test 2022 Q1, Train 2021-2022 Q1 -> Test 2022 Q2) to validate out-of-sample robustness.
3. **Stationarity Enforcement:** Raw prices are banned from the state space. The system must only digest Log Returns or Fractionally Differentiated series to combat concept drift.
4. **Local Poisoning Defense:** The FL Server must score local gradients. Any client returning weights that crash the validation loss on the server side will be dynamically dropped from the aggregation round.

---

## 7. FUTURE DEVELOPMENT PATHS (ROADMAP)

### Phase 1: Core Foundation (Current)

* Build the modular OOP architecture (Data, Env, Agent, Server).
* Implement strict Anti-Leakage Feature Engineering.
* Develop the Event-Driven Backtester with a simulated Slippage/Fee model.

### Phase 2: Algorithmic Complexity

* Transition from `FedAvg` to `FedProx` on the Server side.
* Integrate Prioritized Experience Replay to ensure the AI remembers "Black Swan" crashes.
* Deploy Zero-Lag Kalman Filters to replace standard lagging MAs.

### Phase 3: Multi-Asset & Statistical Arbitrage

* Expand the state space to include Cointegration spreads for Pairs Trading (e.g., BTC/ETH divergence).
* Implement advanced dynamic position sizing (Kelly Criterion).

### Phase 4: Production & Live Execution

* Deploy WebSocket Observers for Live Paper Trading.
* Implement Execution Algorithms (TWAP/VWAP) to minimize slippage on large order entries.
* Activate Hard Circuit Breakers and real-time latency monitoring.


---

## 8. MLOps & INFRASTRUCTURE ARCHITECTURE (DEPLOYMENT)

To transition from a local script to a scalable production system, FedHedge must implement robust Machine Learning Operations (MLOps) and infrastructure standards.

* **Containerization (Docker):** The entire environment must be containerized. The FL Server, FL Clients, and Data Pipelines will run in isolated Docker containers (`docker-compose`) to eliminate the "it works on my machine" problem and ensure environment parity across different operating systems.
* **Experiment Tracking (MLflow / Weights & Biases):** RL and FL training generates thousands of metrics. Using standard console prints is insufficient. The system must integrate an experiment tracker (e.g., MLflow) to log:
* Hyperparameters (Learning rate, PPO clip range, FedProx $\mu$ parameter).
* Training loss graphs.
* Cross-client weight distributions.
* Artifacts (saving the `.pth` checkpoints automatically to a registry).


* **Time-Series Database (TSDB):** For live trading and extensive backtesting, reading from `.csv` files becomes a massive I/O bottleneck. The system will migrate to a dedicated TSDB like **TimescaleDB** (PostgreSQL extension) or **InfluxDB**. This allows for lightning-fast querying of specific timeframes and automatic aggregation of tick data into OHLCV.
* **Asynchronous Processing:** Live data fetching and order execution must use asynchronous Python (`asyncio`, `aiohttp`, `websockets`) to ensure the main trading loop is never blocked waiting for an exchange server to respond.

---

## 9. COMPREHENSIVE EVALUATION METRICS (THE QUANT SCORECARD)

The backtester will output a standardized tear sheet (report) upon completion. The system is evaluated strictly on the following metrics, overriding standard Machine Learning metrics (like MSE or Accuracy).

* **Maximum Drawdown (MDD):** The most critical metric. Measures the largest single drop from peak to bottom in the portfolio's value.
* *System Constraint:* MDD must not exceed $-15\%$ under any backtest scenario.


* **Calmar Ratio:** Measures return relative to drawdown risk.

$$Calmar = \frac{Annualized\_Return}{Maximum\_Drawdown}$$


* *Target:* A Calmar ratio $> 2.0$.


* **Information Ratio (IR):** Measures the active return of the AI's portfolio divided by the tracking error relative to a benchmark (e.g., simply holding Bitcoin/Buy-and-Hold). It proves whether the AI is actually generating Alpha or just riding a bull market.
* **Profit Factor:** The ratio of gross profit to gross loss. Must be $> 1.5$ to account for unforeseen real-world slippage.
* **Average Holding Period:** Tracked to ensure the RL agent is adhering to the Medium-term mandate (1-2 weeks) and not decaying into an overtrading HFT bot.

---

## 10. SECURITY, API MANAGEMENT & FAILSAFES

Handling capital requires bank-grade security protocols at the code level.

* **Credential Isolation:** API keys and Secret keys must **never** be hardcoded or pushed to version control (GitHub). They must be injected at runtime via `.env` files or secure secret managers (e.g., HashiCorp Vault, AWS Secrets Manager).
* **Least Privilege Principle:** * *Training/Backtesting phase:* The system only requires Public Data APIs (no authentication needed).
* *Live Execution phase:* API keys must be strictly generated with **Trade Only** permissions (no Withdrawal permissions) and locked to the specific static IP address of the deployment server (IP Whitelisting).


* **Exponential Backoff & Error Handling:** Exchange APIs frequently fail, return `HTTP 502 Bad Gateway`, or trigger `HTTP 429 Too Many Requests` (Rate Limits). The data pipeline and execution modules must implement automated retry logic with exponential backoff (waiting 1s, then 2s, then 4s) to prevent system crashes during exchange maintenance or extreme volatility.
* **Orphaned Position Failsafe:** If the system crashes, loses power, or loses internet connection while holding a leveraged derivative position, it creates immense risk. The system must implement a "Heartbeat" protocol or utilize exchange-side "Cancel Only" / "Time in Force" settings to ensure orders do not sit infinitely in the order book if the AI goes offline.

---

## 11. DATA PIPELINE & ETL (EXTRACT, TRANSFORM, LOAD) STANDARDS

The system is only as good as the data it consumes.

* **Data Integrity Auditing:** Before any FL training round begins, the ETL pipeline must run an automated audit script to check for:
* *Missing Candles:* Detecting jumps in timestamps.
* *Zero Volume Anomalies:* Identifying periods where the exchange API returned dead data.
* *Price Spikes:* Filtering out impossible API glitches (e.g., BTC dropping to $1 for a single tick).


* **Synthetic Data Generation (Data Augmentation):** To train a robust RL agent, historical data alone may not contain enough "Black Swan" events. The pipeline will eventually include tools to generate synthetic market crashes (using Geometric Brownian Motion or GANs) to stress-test the RL agent's hedging capabilities during training.
* **Continuous Updating (Cron Jobs):** The historical database must be updated automatically daily via schedulers (e.g., Apache Airflow or simple CRON) so the RL model can be retrained periodically on the freshest market regimes.


---

## 12. EXECUTION STATE MACHINE (ORDER LIFECYCLE MANAGEMENT)

To prevent ghost orders or double-spending, the live trading module must operate as a strict Finite State Machine (FSM). The AI cannot simply "send an order and forget." Every action must pass through 4 distinct states.

* **State 1: Signal Generation:** The RL Agent outputs a target hedge ratio (e.g., $h = 0.4$).
* **State 2: Pre-Trade Risk Check:** The Execution Engine intercepts the signal and checks hard constraints. It verifies available margin, checks if the daily drawdown limit is breached, and ensures the new position does not exceed maximum leverage rules.
* **State 3: Algorithmic Routing:** If approved, a large order is not sent as a single Market Order. It is passed to an execution algorithm (TWAP - Time Weighted Average Price) that slices the order into smaller chunks over a 5-15 minute window to minimize market impact and slippage.
* **State 4: Post-Trade Reconciliation:** Once executed, the system queries the exchange API via WebSockets to confirm the exact filled price and volume. The internal portfolio state is updated strictly based on the exchange's confirmation, not the AI's assumption.

---

## 13. CONTINUOUS LEARNING & RETRAINING PIPELINE

The cryptocurrency market exhibits constant concept drift. A model trained on 2023 data will decay in performance by 2025. The system requires an automated continuous learning pipeline.

* **Trigger Mechanisms:** Retraining is not random. It is triggered either by time (e.g., every 14 days) or by performance decay (e.g., if the rolling Sharpe ratio drops below 1.0 for a week).
* **Warm Starting:** When retraining, the RL agent does not start from scratch. It loads the latest `.pth` checkpoint (Warm Start) and updates its weights using only the most recent market data, preserving its long-term memory while adapting to the new micro-regime.
* **Shadow Deployment (Canary Testing):** The newly trained model is not immediately given capital. It is deployed in a "Shadow Mode" alongside the live model. It processes real-time data and generates hypothetical trades. If the Shadow Model outperforms the Live Model over a 7-day validation period, an automated hot-swap occurs.

---

## 14. DISASTER RECOVERY & BUSINESS CONTINUITY PLAN (BCP)

Institutional systems must survive infrastructure failures.

* **Database Redundancy:** All historical data and trade logs must be replicated to a secondary storage bucket (e.g., AWS S3 or a secondary local hard drive) daily.
* **Graceful Degradation:** If the Central FL Server goes offline, the local RL clients on the exchange nodes must not crash. They must switch to a "Standalone Mode," continuing to trade using their last known local weights, while actively attempting to reconnect to the server via exponential backoff.
* **Kill Switch Automation:** A physical or web-based manual "Kill Switch" must be wired directly to the exchange APIs. If the server is compromised or the AI acts erratically, a human operator can hit the switch to instantly send "Close All Positions" and "Revoke API Keys" commands, bypassing the Python system entirely.
