# Machine Learning Integrated Tail Risk Detection

**Using GARCH, Extreme Value Theory, and Gradient Boosting for Financial Crisis Prediction**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Quantitative%20Finance-orange)](https://github.com/yourusername/tail-risk-detection)

## Overview

This project introduces a novel framework that integrates econometric models (GARCH volatility dynamics and Extreme Value Theory) with machine learning to predict tail risk events in financial markets. Rather than asking *"What is today's VaR?"* (estimation), we reframe the problem as *"Will tomorrow's return exceed VaR?"* (prediction), enabling early detection of regime shifts.

### Key Results

- **60.1% AUC** on 2008 financial crisis data
- **73.5% AUC** on out-of-sample COVID-19 crash (+22.3% improvement)
- **GARCH volatility dominates predictions** (21.8% feature importance)
- **Novel interaction features** capture systemic stress (14.1% importance)

## Motivation

Traditional Value-at-Risk (VaR) models systematically failed during the 2008 financial crisis due to:

1. **Static volatility assumptions** - Ignoring volatility clustering
2. **Gaussian distributions** - Underweighting fat-tail probabilities  
3. **Backward-looking calibration** - Cannot anticipate regime shifts
4. **Methodological isolation** - Missing complementary information across models

This framework addresses these limitations by creating a unified early warning system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
│  5 Major U.S. Banks (JPM, GS, MS, C, BAC) → Daily Returns   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering (40+ Features)            │
├─────────────────────────────────────────────────────────────┤
│  • Rolling Statistics (mean, std, skew, kurtosis)            │
│  • GARCH Dynamics (conditional vol, leverage effects)        │
│  • EVT Tail Measures (GPD shape, scale, exceedance prob)     │
│  • Cross-Sectional Stress (correlation, dispersion)          │
│  • Interaction Features (vol × dispersion) [NOVEL]           │
│  • Temporal Patterns (exceedance clustering)                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              XGBoost Classification Model                     │
│  Binary Target: Will return exceed 95% VaR threshold?        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│         Output: Tail Risk Probability + SHAP Values          │
│  • Early warning signals for risk committees                 │
│  • Feature importance for scenario analysis                  │
│  • Model validation via backtesting                          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
tail-risk-detection/
├── data/
│   ├── raw/                      # Raw price data from Yahoo Finance
│   ├── processed/                # Cleaned returns and features
│   └── profiles/                 # Saved EVT parameter profiles
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_garch_modeling.ipynb
│   ├── 03_evt_estimation.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_ml_training.ipynb
│   └── 06_backtesting.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Yahoo Finance data acquisition
│   │   └── portfolio.py         # Equal-weighted portfolio construction
│   ├── models/
│   │   ├── garch.py             # GARCH(1,1) and GJR-GARCH estimation
│   │   ├── evt.py               # Peaks-Over-Threshold EVT fitting
│   │   └── var_models.py        # Traditional VaR (parametric, historical, MC)
│   ├── features/
│   │   ├── rolling_stats.py     # Rolling window statistics
│   │   ├── garch_features.py    # GARCH-derived features
│   │   ├── evt_features.py      # EVT-derived features
│   │   ├── cross_sectional.py   # Cross-sectional stress indicators
│   │   └── interactions.py      # Interaction feature construction
│   ├── ml/
│   │   ├── classifier.py        # XGBoost training with walk-forward validation
│   │   ├── evaluation.py        # Performance metrics and ROC curves
│   │   └── interpretation.py    # SHAP value analysis
│   ├── backtesting/
│   │   ├── kupiec_test.py       # Kupiec Proportion of Failures test
│   │   └── christoffersen_test.py # Christoffersen Independence test
│   └── utils/
│       ├── plotting.py          # Visualization utilities
│       └── helpers.py           # Common helper functions
├── tests/
│   └── test_*.py                # Unit tests for all modules
├── results/
│   ├── figures/                 # Generated plots and charts
│   ├── metrics/                 # Performance metrics CSV files
│   └── models/                  # Saved model checkpoints
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## Getting Started

### Prerequisites

```bash
Python 3.8+
NumPy, Pandas, SciPy
Scikit-learn, XGBoost, SHAP
Matplotlib, Seaborn
yfinance (for data acquisition)
```


### Quick Start

```python
from src.data.data_loader import load_bank_data
from src.models.garch import fit_garch
from src.models.evt import fit_evt
from src.features.feature_engineering import create_features
from src.ml.classifier import train_tail_risk_model

# 1. Load data for 5 major banks
tickers = ['JPM', 'GS', 'MS', 'C', 'BAC']
data = load_bank_data(tickers, start='2005-01-01', end='2010-12-31')

# 2. Fit GARCH model
garch_params, cond_vol = fit_garch(data['returns'])

# 3. Fit EVT model
evt_params = fit_evt(data['returns'], threshold_percentile=90)

# 4. Engineer features
features = create_features(data, garch_params, evt_params)

# 5. Train XGBoost classifier
model, metrics = train_tail_risk_model(features, walk_forward=True)

print(f"In-Sample AUC: {metrics['auc']:.3f}")
print(f"Top Features: {metrics['feature_importance'][:5]}")
```

## 📈 Key Features

### 1. GARCH Volatility Modeling
- **GARCH(1,1)**: Captures volatility clustering and persistence
- **GJR-GARCH**: Models leverage effects (asymmetric volatility response)
- **Conditional Variance**: Forward-looking volatility estimates
- **Volatility of Volatility**: Second-order dynamics

### 2. Extreme Value Theory
- **Peaks-Over-Threshold (POT)**: Focus on tail exceedances
- **Generalized Pareto Distribution**: Heavy-tailed distribution fitting
- **Time-Varying Parameters**: Rolling 60-day window estimates
- **Mean Residual Life Plots**: Threshold selection diagnostics

### 3. Feature Engineering
40+ features across 6 categories:
- Rolling statistics (mean, std, skewness, kurtosis)
- GARCH dynamics (conditional vol, persistence, leverage)
- EVT tail measures (shape, scale, exceedance probability)
- Cross-sectional stress (correlation, dispersion)
- **Novel interactions** (vol × dispersion, GARCH × EVT)
- Temporal patterns (exceedance clustering, days since last event)

### 4. Machine Learning Pipeline
- **XGBoost Classifier**: Gradient boosting with regularization
- **Walk-Forward Validation**: Respects temporal structure
- **Class Imbalance Handling**: Scale_pos_weight adjustment
- **SHAP Interpretability**: Game-theoretic feature attribution

### 5. Backtesting Framework
- **Kupiec Test**: Proportion of failures validation
- **Christoffersen Test**: Independence of exceedances
- **Multiple VaR Methods**: Parametric, Historical, Monte Carlo

## 📊 Results Summary

### Performance Metrics

| Period | Dataset | AUC-ROC | Precision | Recall | F1-Score |
|--------|---------|---------|-----------|--------|----------|
| 2005-2010 | In-Sample (Overall) | 0.601 | 0.189 | 0.673 | 0.295 |
| 2007-2009 | In-Sample (Crisis) | 0.642 | 0.180 | 0.559 | 0.273 |
| 2019-2020 | Out-of-Sample (COVID) | **0.735** | 0.243 | 0.741 | 0.366 |

**Key Insight**: +22.3% performance improvement on out-of-sample COVID-19 data demonstrates cross-regime generalization.

### Feature Importance (SHAP Values)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | GARCH Conditional Volatility | 21.8% | GARCH |
| 2 | Volatility × Dispersion | 14.1% | **Novel Interaction** |
| 3 | 20-Day Rolling Volatility | 7.3% | Traditional |
| 4 | Volatility of Volatility | 7.1% | GARCH |
| 5 | EVT Shape Parameter | 7.1% | EVT |

**Key Finding**: GARCH volatility provides 197% improvement over rolling windows, validating dynamic modeling approach.

### Structural Break Validation

| Metric | Pre-Crisis | Crisis | Increase |
|--------|------------|--------|----------|
| Annualized Volatility | 14.8% | 85.6% | **5.8×** |
| Average Correlation | 0.58 | 0.81 | +40% |
| Excess Kurtosis | 1.23 | 8.74 | **7.1×** |
| Chow Test F-Statistic | - | 33.5 | p < 10⁻¹⁰ |

## 🔬 Methodology Details

### GARCH(1,1) Specification

The return process and conditional variance equation:

```
r_t = μ + ε_t
ε_t = σ_t × z_t,  z_t ~ N(0,1)
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

Where:
- **ω**: Long-run variance base level
- **α**: ARCH coefficient (shock response)
- **β**: GARCH coefficient (persistence)
- **α + β ≈ 1**: High persistence (volatility clustering)

Estimated via Maximum Likelihood with constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1.

### GJR-GARCH Extension

Captures leverage effects (asymmetric volatility):

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·ε²_{t-1}·I_{t-1}
I_{t-1} = 1 if ε_{t-1} < 0, else 0
```

**γ > 0**: Negative shocks increase volatility more than positive shocks.

### Extreme Value Theory (EVT)

**Peaks-Over-Threshold (POT)** models exceedances above threshold u:

```
P(X > u + y | X > u) ≈ GPD(y; ξ, σ)
GPD(y; ξ, σ) = 1 - (1 + ξy/σ)^{-1/ξ}
```

Where:
- **ξ (xi)**: Shape parameter (tail index)
  - ξ > 0: Heavy tails (power law) ← Financial data
  - ξ = 0: Exponential tails
  - ξ < 0: Bounded tails
- **σ (sigma)**: Scale parameter

**Threshold Selection**: 90th percentile of absolute returns (~1.5%), validated via Mean Residual Life plots.

### XGBoost Configuration

```python
xgb_params = {
    'learning_rate': 0.1,        # Conservative learning
    'max_depth': 3,              # Shallow trees (interpretability)
    'n_estimators': 100,         # Number of boosting rounds
    'subsample': 0.8,            # Stochastic gradient boosting
    'colsample_bytree': 0.8,     # Feature sampling
    'scale_pos_weight': n_neg/n_pos,  # Class imbalance handling
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
```

**Walk-Forward Validation**: Quarterly retraining (every 60 days) using 250-day rolling window.

## Visualizations

### 1. GARCH Conditional Volatility

Shows dynamic volatility estimation capturing crisis regime shifts:

```python
from src.utils.plotting import plot_garch_volatility

plot_garch_volatility(returns, cond_vol, crisis_periods)
# Peaks during Jul 2007-Mar 2009 (2008 crisis) and Mar 2020 (COVID-19)
```

### 2. Rolling Correlation Breakdown

Demonstrates diversification collapse during crises:

```python
from src.utils.plotting import plot_rolling_correlation

plot_rolling_correlation(stock_returns, window=60)
# Shows surge from 0.58 → 0.81 during crisis
```

### 3. Tail Risk Probability Over Time

Model predictions with actual tail events:

```python
from src.ml.evaluation import plot_predictions

plot_predictions(model, test_data, threshold=0.1)
# Elevated probabilities preceding major drawdowns
```

### 4. SHAP Feature Importance

Waterfall plot showing feature contributions:

```python
from src.ml.interpretation import plot_shap_importance

plot_shap_importance(model, features, top_k=15)
# GARCH conditional vol dominates at 21.8%
```

## Backtesting

### Kupiec Proportion of Failures Test

Tests if observed VaR exceedances match theoretical level:

```python
from src.backtesting.kupiec_test import kupiec_test

result = kupiec_test(returns, var_estimates, confidence=0.95)
print(f"LR Statistic: {result['lr_stat']:.2f}, p-value: {result['p_value']:.4f}")
# Crisis period: 141 violations vs. 25.2 expected (p < 0.001) → REJECT
```

**Finding**: All traditional VaR methods systematically fail during 2007-2009, experiencing 3-5× more exceedances than predicted.

### Christoffersen Independence Test

Tests if exceedances are independent (no clustering):

```python
from src.backtesting.christoffersen_test import christoffersen_test

result = christoffersen_test(returns, var_estimates, confidence=0.95)
print(f"Independence rejected: {result['reject']}")
# Crisis period: p < 0.01 → Violations are clustered
```

## Applications

### 1. Early Warning System
**Use Case**: Risk committees monitoring regime shifts
- Elevated predicted probabilities (>30%) trigger deeper analysis
- Addresses VaR's backward-looking limitation
- Example: March 2020 probabilities averaged 38% vs. 15% baseline

### 2. Model Risk Management
**Use Case**: Identifying when traditional VaR models fail
- When Historical Simulation and Parametric VaR diverge, ML flags potential inadequacy
- Provides independent validation layer

### 3. Dynamic Hedging
**Use Case**: Adjusting protective option positions
- Predicted tail probability informs hedge ratios
- 2× probability increase → proportional hedge adjustment
- Cost-benefit analysis required (option premiums vs. protection value)

### 4. Stress Testing
**Use Case**: Scenario generation for risk committees
- SHAP values reveal drivers: "What if GARCH volatility reaches 2008 levels?"
- Trace impact through interaction features
- Quantify compound risk effects

### 5. Derivatives Pricing Extensions
**Use Case**: Volatility surface calibration
- Model identifies regime shifts → adjust implied volatility forecasts
- Example: Option premium increases 10.5× during crisis peaks

## Limitations & Caveats

### Critical Limitations

1. **Survivorship Bias** (Most Critical)
   - Excludes Lehman Brothers and Bear Stearns (bankruptcy)
   - All results are **conditional on survival**
   - True unconditional tail risk likely more severe
   - Analogous to studying plane crashes by interviewing only survivors

2. **False Alarm Rate**
   - Precision: 18.9% → 81% of warnings are false positives
   - **Necessitates use as screening tool, not automated signal**
   - Cost of investigation must be acceptable

3. **Single Crisis Training**
   - Trained only on 2008 crisis
   - COVID-19 validation encouraging but limited
   - Future crises may exhibit novel patterns

4. **Probability Calibration**
   - Raw XGBoost probabilities not well-calibrated
   - Platt scaling or isotonic regression needed for precise probabilities
   - Current use: **relative rankings, not absolute probabilities**

5. **Non-Stationarity**
   - Financial markets evolve (regulations, structure, macro regimes)
   - Quarterly retraining helps but cannot anticipate unprecedented events
   - Recommend continuous monitoring and recalibration

### Practical Constraints

- **Liquidity/Transaction Costs**: Assumes frictionless markets
- **Feature Engineering Bias**: 40+ features reflect researcher judgment
- **Computational Cost**: GARCH/EVT estimation + XGBoost training requires ~5-10 minutes per quarterly retrain

## Future Work

### Methodological Extensions

1. **Copula-Based Tail Dependence**
   - Model asymmetric correlation structure during crises
   - Capture joint tail behavior beyond pairwise correlations

2. **Regime-Switching GARCH**
   - Explicit transition probabilities between volatility regimes
   - Sharper detection of regime shifts

3. **Alternative Data Integration**
   - VIX term structure (implied volatility expectations)
   - CDS spreads (credit risk indicators)
   - Sentiment indicators (news analytics, social media)

4. **Deep Learning Architectures**
   - LSTM for temporal dependency modeling
   - Attention mechanisms for feature weighting
   - Autoencoder for unsupervised feature learning

5. **Network-Based Systemic Risk**
   - Granger causality networks
   - Graph neural networks for contagion modeling

### Empirical Extensions

1. **Multi-Asset Validation**
   - Credit markets (corporate bonds, CDS)
   - Commodities (oil, metals)
   - Foreign exchange rates

2. **Real-Time Deployment**
   - Streaming data pipelines
   - Online learning with incremental updates
   - Production-grade monitoring dashboard

3. **Counterfactual Analysis**
   - "What if model had been deployed in 2007?"
   - Quantify value of early warnings via hedging P&L simulation

## References

### Core Papers

1. **GARCH Models**
   - Engle (1982): Autoregressive Conditional Heteroskedasticity
   - Bollerslev (1986): Generalized ARCH
   - Glosten et al. (1993): GJR-GARCH leverage effects

2. **Extreme Value Theory**
   - Balkema & de Haan (1974): Residual life theorem
   - Pickands (1975): Statistical inference using extreme order statistics
   - McNeil & Frey (2000): EVT for heteroskedastic time series

3. **Machine Learning in Finance**
   - Chen & Guestrin (2016): XGBoost framework
   - Lundberg & Lee (2017): SHAP values
   - Gu, Kelly & Xiu (2020): Empirical asset pricing via ML

4. **Risk Management**
   - Basel Committee (2004): Capital measurement standards
   - Christoffersen (1998): Evaluating interval forecasts
   - Berkowitz et al. (2011): Evaluating VaR with desk-level data

### Complete Bibliography
See `references.bib` for full list of 30+ papers cited.

## Contributing

Contributions welcome! Areas of particular interest:

1. **Alternative Feature Sets**: Novel econometric features or interaction terms
2. **Additional Crises**: Dot-com bubble (2000-2002), European debt crisis (2010-2012)
3. **Deployment Tools**: Streamlit dashboard, real-time monitoring
4. **Optimization**: Faster GARCH estimation, GPU-accelerated XGBoost

**Process**:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes with descriptive messages
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request with detailed description


## 👤 Author

**Maksim Silchenko**  
BSc International Business, Bayes Business School, City, University of London  
📧 maksim.silchenko@bayes.city.ac.uk  

**Research Interests**: Quantitative Finance, Machine Learning, Risk Management, Financial Engineering

## 🙏 Acknowledgments

- Bayes Business School faculty for guidance on quantitative finance methodologies
- Open-source community for Python scientific computing stack
- Yahoo Finance for accessible financial data API
- Reviewers and contributors for feedback and improvements

## 📊 Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{silchenko2024tailrisk,
  title={Machine Learning Integrated Tail Risk Detection Using GARCH, Extreme Value Theory, and Gradient Boosting},
  author={Silchenko, Maksim},
  institution={Bayes Business School, City, University of London},
  year={2024},
  note={GitHub: https://github.com/yourusername/tail-risk-detection}
}
```

---

**⭐ If you find this project useful, please consider starring the repository!**

**📢 Questions or feedback?** Open an issue or reach out via email.

**🔔 Stay Updated**: Watch the repository for new features and improvements.
