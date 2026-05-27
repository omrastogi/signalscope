# SignalScope

Learning AI Adoption Signals in an Automated Trading System

---

## Overview

SignalScope is a reinforcement learning based trading research project that studies whether publicly available AI adoption signals can help predict short-term stock market behavior around AI-related events.

The system combines:

- Google Trends AI interest signals
- Earnings call AI keyword frequency
- Stock price and volume behavior

to train a Q-learning agent that learns a simple:

- Buy
- Hold
- Sell

policy for technology sector stocks.

The project is designed as a lightweight and interpretable alternative to institutional quantitative systems, using only freely accessible public data sources.

---

# Project Goal

Retail investors often react late to AI-related announcements.

This project explores whether observable public signals such as:

- spikes in AI-related searches,
- increased discussion of AI in earnings calls,
- abnormal trading activity,

can be transformed into actionable trading signals.

The reinforcement learning agent attempts to:

- maximize short-term portfolio returns,
- reduce excessive drawdowns,
- learn trading behavior around AI adoption events.

---

# Initial Scope

The first version focuses on:

- Tabular Q-Learning
- Discrete state spaces
- Tech-sector stocks
- Publicly available datasets
- Simple risk management

No deep learning or advanced RL libraries will be used initially.

---

# Repository Structure

```text
SignalScope/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── eda/
│   └── experiments/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── environment/
│   ├── rl/
│   ├── backtesting/
│   └── utils/
│
├── reports/
│
├── configs/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# Dataset Sources

## 1. Stock Market Data

### Source

- Yahoo Finance

### Python Library

- `yfinance`

### Data Collected

- Open price
- Close price
- High
- Low
- Volume
- Adjusted close

### Example Stocks

- NVDA
- MSFT
- META
- AMZN
- GOOG
- AAPL

### Example Download

```python
import yfinance as yf

df = yf.download(
    "NVDA",
    start="2021-01-01",
    end="2025-01-01"
)

print(df.head())
```

---

# 2. Google Trends Data

## Source

- Google Trends

## Python Library

- `pytrends`

## Signals

Example keywords:

- artificial intelligence
- generative ai
- ai stocks
- openai
- chatgpt

## Example Download

```python
from pytrends.request import TrendReq

pytrends = TrendReq()

kw_list = ["generative ai"]

pytrends.build_payload(
    kw_list,
    timeframe='2021-01-01 2025-01-01'
)

df = pytrends.interest_over_time()

print(df.head())
```

---

# 3. SEC Filing / Earnings Data

## Source

- SEC EDGAR
EDGAR is the filing system used by the U.S. Securities and Exchange Commission.
Public companies in the US are legally required to submit documents there. Think of it as: the official database of company disclosures.

## Data Collected

- 10-K filings
- 10-Q filings
- earnings call transcripts
- AI-related keyword frequency

## Example Keywords

- artificial intelligence
- machine learning
- automation
- generative AI
- foundation model
- LLM

## Initial Approach

Simple keyword frequency counts:

- frequency per filing
- normalized keyword frequency
- rolling change in AI mentions

No LLMs initially.

---

# Initial Dataset Schema

```text
date
ticker
close
volume
daily_return
5d_forward_return
10d_forward_return
volume_zscore
ai_trend_score
ai_keyword_frequency
drawdown_5d
```

---

# Phase 1: Exploratory Data Analysis (EDA)

Before building the RL system, we first validate whether meaningful signal exists.

## EDA Goals

### Market Behavior

- How do stocks behave after AI-related signal spikes?
- Are returns unusually volatile?

### Signal Analysis

- Do Google Trends spikes align with volume spikes?
- Do AI mentions increase before large price moves?

### Risk Analysis

- What are the drawdowns after signal events?
- Are some stocks more stable than others?

---

# Planned EDA Visualizations

- Stock price vs time
- Volume spike analysis
- Google Trends over time
- Correlation heatmaps
- Return distributions
- Forward return analysis
- Drawdown analysis
- Event-window volatility

---

# Reinforcement Learning Setup

## State Space

Signals bucketed into:

- Low
- Medium
- High

Example state:

```text
AI Trend = High
Volume Spike = Medium
Recent Return = Low
```

---

# Actions

- Buy
- Hold
- Sell

---

# Reward Function

Portfolio return over:

- 5 trading days
- with drawdown penalty

---

# Risk Management

The system includes:

- stop-loss logic
- drawdown penalties
- volatility-aware thresholds

---

# Tech Stack

## Core

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Data

- yfinance
- pytrends
- SEC EDGAR API

## RL

Implemented from scratch:

- Q-table
- epsilon-greedy exploration
- Bellman updates

---

# Installation

```bash
git clone <repo_url>

cd SignalScope

pip install -r requirements.txt
```

---

# First Milestone

Goal:

Train a basic tabular Q-learning agent using:

- one stock,
- Google Trends signals,
- historical price data,

and compare against:

- Buy-and-Hold
- Random trading strategy

---

# Long-Term Extensions

Potential future improvements:

- Deep Q Networks (DQN)
- Transformer-based filing analysis
- News sentiment
- Portfolio optimization
- Multi-stock RL environments
- Live signal dashboard

---

# Research Motivation

This project explores whether publicly observable AI adoption behavior creates measurable short-term trading inefficiencies that can be exploited using lightweight reinforcement learning methods.

The focus is not high-frequency trading, but interpretable event-driven decision making using public alternative data.