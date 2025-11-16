# ML-Integrated Tail Risk Framework

A quantitative risk management system that transforms traditional econometric models (GARCH, EVT) into machine learning features for predicting financial market tail events. Developed using U.S. investment bank data during the 2008 financial crisis, with out-of-sample validation on the COVID-19 market crash.

## Overview

This project addresses a fundamental limitation in quantitative finance: treating risk models as standalone estimators rather than integrated components of a predictive system. Instead of using GARCH for volatility forecasting and Extreme Value Theory (EVT) for tail risk separately, this framework leverages their outputs as engineered features within a supervised learning pipeline.

The core hypothesis is that regime shifts exhibit learnable patterns—captured through volatility clustering dynamics, tail thickness evolution, and cross-sectional market stress indicators—that can be detected before tail events materialize.

## Key Results

| Metric | Performance |
|--------|------------|
| 2008 Financial Crisis AUC | 0.671 |
| COVID-19 Out-of-Sample AUC | **0.735** |
| Feature Importance (GARCH conditional vol) | 22% |
| Tail Event Detection Rate | 67-74% |

The model's improved performance on the COVID-19 crisis (entirely unseen during training) suggests that learned tail risk patterns transfer across different crisis regimes, validating the generalization capability of the integrated approach.

## Technical Implementation

### Risk Models

**GARCH(1,1) & GJR-GARCH**: Custom maximum likelihood estimation implementation for conditional volatility modeling. Parameters estimated via L-BFGS-B optimization with stationarity constraints.

```python
σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
```

The GJR variant captures leverage effects (asymmetric volatility response to positive/negative shocks), which proved important for crisis detection.

**Extreme Value Theory (Peaks-Over-Threshold)**: Rolling estimation of Generalized Pareto Distribution parameters to track tail thickness evolution. The shape parameter ξ serves as a dynamic measure of tail risk.

### Feature Engineering Pipeline

The framework constructs 70+ features across five categories:

1. **GARCH-Based Dynamics** (22% total importance)
   - Conditional volatility levels and momentum
   - Volatility-of-volatility (second-order risk)
   - Forecast error magnitude (regime shift indicator)
   - Z-score normalized volatility

2. **EVT Tail Parameters**
   - Rolling GPD shape parameter (60/120-day windows)
   - Tail risk momentum and acceleration
   - Shape parameter z-scores

3. **Cross-Sectional Stress Indicators**
   - Return dispersion across portfolio constituents
   - Dynamic correlation structure (breakdown detection)
   - Correlation regime change velocity

4. **Novel Interaction Terms**
   - GARCH volatility × EVT shape (compound tail risk)
   - Dispersion × Volatility interaction (systemic stress signal)
   - Multi-indicator warning score

5. **Temporal Risk Patterns**
   - Recent exceedance frequency
   - Days since last tail event
   - Volatility ratio regimes

### Machine Learning Architecture

**Model**: XGBoost classifier with class imbalance handling (`scale_pos_weight` adjustment for ~5% positive class frequency)

**Validation Strategy**: Walk-forward cross-validation with expanding training window
- Initial training: 504 days (2 years)
- Step size: 63 days (quarterly retraining)
- No look-ahead bias in feature construction

**Interpretability**: SHAP (SHapley Additive exPlanations) values for feature importance decomposition, validating that GARCH/EVT-derived features dominate predictions over simple rolling statistics.

### Backtesting Framework

Rigorous statistical validation using:
- **Kupiec Proportion of Failures Test**: Unconditional coverage testing
- **Christoffersen Independence Test**: Violation clustering detection
- Both tests applied to evaluate VaR model performance across different methodologies

### Derivatives Pricing Extension

The framework extends to regime-aware options pricing:
- Black-Scholes with GARCH-implied volatility
- Crisis volatility (262% annualized) yields option premiums 10.5× higher than historical average
- Demonstrates practical application for hedging cost estimation during regime shifts

## Data

- **Assets**: Major U.S. investment banks (JPM, GS, MS, C, BAC)
- **Training Period**: January 2005 – December 2010 (financial crisis)
- **Out-of-Sample Validation**: February – March 2020 (COVID-19 crash)
- **Frequency**: Daily returns, equal-weighted portfolio

The choice of investment banks during 2008 provides an extreme stress test environment where traditional VaR models systematically failed.

## Repository Structure

```
├── VaR_Analysis_Advanced.ipynb    # Main analysis notebook
├── README.md                      # This file
└── requirements.txt               # Dependencies (if included)
```

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- scikit-learn
- XGBoost
- yfinance
- matplotlib, seaborn

## Usage

```python
# Load and run the complete analysis
jupyter notebook VaR_Analysis_Advanced.ipynb
```

The notebook is structured sequentially:
1. Data acquisition and exploratory analysis
2. GARCH model estimation
3. EVT threshold selection and fitting
4. Feature engineering pipeline
5. Walk-forward ML training
6. Out-of-sample validation
7. SHAP interpretability analysis
8. Derivatives pricing application

## Limitations and Future Work

**Current Limitations**:
- Probability calibration needed for direct hedging execution
- Single asset class (equities) validation
- Limited crisis regime samples for training
- Survivorship bias in historical data (failed institutions excluded)

**Extensions**:
- Multi-asset class validation (credit, commodities, FX)
- Alternative ML architectures (LSTM for temporal patterns)
- Real-time feature computation optimization
- Ensemble methods combining multiple risk models
- Transaction cost integration for trading signals

## Practical Applications

This framework is designed as a **risk committee screening tool** rather than an automated trading system. Key use cases:

- Early warning system for tail risk regime shifts
- Stress testing and scenario analysis
- Dynamic VaR model selection
- Hedging cost estimation during volatility regimes
- Model risk management (identifying when traditional VaR fails)

## Technical Notes

The GARCH and EVT implementations are built from first principles rather than using library wrappers, demonstrating understanding of:
- Maximum likelihood estimation under constraints
- Numerical optimization for financial models
- Statistical testing for model validation
- Feature engineering for time series classification

This approach prioritizes interpretability and pedagogical clarity over production optimization.


*This project was developed as part of independent research into quantitative risk management methodologies, combining classical financial econometrics with modern machine learning techniques.*
