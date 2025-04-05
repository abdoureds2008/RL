#  UltimateRLTrading

A fully modular and robust framework for reinforcement learning-based algorithmic trading and optimal execution strategies.

---

##  Key Features

 Multi-timeframe data support (daily & intraday)  
 Integrated fundamentals and sentiment analysis  
 Short selling, partial fills, limit orders, churn cost, short borrow fees  
 Customizable agents:  
- Double DQN with LSTM  
- PPO (Stable-Baselines3)  
- Ensemble Agent (majority voting)  
 Reward shaping: VaR, CVaR, drawdown penalties  
 SHAP interpretability and TensorBoard logging  
 Hyperparameter optimization with Optuna  
 GPU usage (TensorFlow)  
Live/paper-trading mode & retraining stubs

---

## Project Structure
```text
UltimateRLTrading/
├── agents/                  # RL agents (DQN, PPO, Ensemble)
│   ├── __init__.py
│   ├── double_dqn.py
│   ├── ensemble.py
│   └── ppo_agent.py
│
├── data/                    # Data loading utilities
│   ├── __init__.py
│   └── loader.py
│
├── envs/                    # Custom multi-asset trading environment
│   ├── __init__.py
│   └── multi_asset_env.py
│
├── evaluation/              # Evaluation logic (episode play & walkforward)
│   ├── __init__.py
│   ├── play.py
│   └── walkforward.py
│
├── Features/                # Feature engineering
│   └── Features.py
│
├── models/                  # Neural network models (LSTM, etc.)
│   ├── __init__.py
│   └── lstm_model.py
│
├── utils/                   # Logging, replay buffer, scalers, etc.
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   ├── Optimisation.py
│   ├── Replay.py
│   ├── retraining.py
│   ├── scaler.py
│   └── tensorboard_tools.py
│
├── main.py                  # Main training/testing entrypoint
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---


##  How to Use

### Train a discrete agent on daily data:

```bash
python main.py --mode train --agent discrete --resolution daily
```
### Test a PPO agent on 1-minute data:

```bash
python main.py --mode test --agent continuous --resolution 1min
```

### Perform walk-forward cross-validation:

```bash
python main.py --mode walkforward
```

### Run hyperparameter optimization (Optuna):

```bash
python main.py --mode train --optimize
```
###  Visualize in TensorBoard

```bash
tensorboard --logdir=logs/
```
## Dependencies
### Install all required packages:
```bash
pip install -r requirements.txt
```
## Sample Input Data:

Ensure you have the following .csv files (with date column):

aapl_msi_sbux.csv (daily prices)

intraday_prices_1min.csv (optional)

fundamentals.csv (optional)

sentiment.csv (optional)

## Author
Zahir Abdellah – Modular RL Trading System