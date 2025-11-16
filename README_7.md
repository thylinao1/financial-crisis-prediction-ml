# ML-Integrated Tail Risk Framework

A quantitative risk management system that transforms traditional econometric models (GARCH, EVT) into machine learning features for predicting financial market tail events. Developed using U.S. investment bank data during the 2008 financial crisis, with out-of-sample validation on the COVID-19 market crash.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Motivation and Research Question](#motivation-and-research-question)
3. [Key Results](#key-results)
4. [Data and Portfolio Construction](#data-and-portfolio-construction)
5. [Methodology](#methodology)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [GARCH Volatility Modeling](#garch-volatility-modeling)
   - [Extreme Value Theory](#extreme-value-theory)
   - [VaR Methodologies Comparison](#var-methodologies-comparison)
   - [Machine Learning Pipeline](#machine-learning-pipeline)
   - [Backtesting Framework](#backtesting-framework)
6. [Feature Engineering Architecture](#feature-engineering-architecture)
7. [Model Validation](#model-validation)
8. [Derivatives Pricing Extension](#derivatives-pricing-extension)
9. [Critical Limitations](#critical-limitations)
10. [Repository Structure](#repository-structure)
11. [Installation and Usage](#installation-and-usage)
12. [Future Work](#future-work)
13. [References](#references)

---

## Executive Summary

This project addresses a fundamental limitation in quantitative risk management: treating econometric models as standalone estimators rather than integrated components of a predictive system. The framework transforms GARCH volatility dynamics and Extreme Value Theory parameters into engineered features for supervised learning, creating an early warning system for tail risk regime shifts.

**Core Innovation**: Instead of asking "What is today's VaR?" (estimation), we ask "Will tomorrow's return exceed VaR?" (prediction). This reframing enables integration of multiple risk signals into a unified detection system.

| Metric | Result |
|--------|--------|
| In-Sample AUC (2008 Crisis) | 0.671 |
| Out-of-Sample AUC (COVID-19) | **0.735** |
| Top Feature (SHAP) | GARCH conditional volatility (22%) |
| Novel Contribution | Dispersion × Volatility interaction (#2 feature) |
| Tail Event Detection Rate | 67-74% |

The improved COVID-19 performance (entirely unseen during training) validates that learned tail risk patterns transfer across different crisis regimes—the model captures something fundamental about how extreme events manifest, not just idiosyncrasies of 2008.

---

## Motivation and Research Question

Traditional Value-at-Risk models failed catastrophically during the 2008 financial crisis. Banks using historical simulation or parametric VaR systematically underestimated tail risk, partly because:

1. **Static volatility assumptions** ignored volatility clustering
2. **Normal distribution assumptions** underweighted extreme events
3. **Backward-looking calibration** couldn't anticipate regime shifts
4. **Isolated methodologies** didn't leverage complementary information

**Research Question**: Can we combine the strengths of multiple risk paradigms—GARCH's dynamic volatility, EVT's tail focus, and ML's pattern recognition—into an integrated system that detects regime shifts before they fully materialize?

**Hypothesis**: Regime shifts exhibit learnable patterns through:
- Volatility clustering acceleration (GARCH)
- Tail thickness evolution (EVT shape parameters)
- Cross-sectional stress propagation (correlation breakdown)
- Compound risk interactions (volatility × tail risk)

---

## Key Results

### Quantitative Findings

1. **GARCH Features Dominate Predictions**
   - Conditional volatility: 22% feature importance (SHAP validated)
   - 3× more predictive than traditional rolling volatility
   - Captures regime dynamics that static windows miss

2. **Model Generalizes Across Crises**
   - Training: 2005-2010 (Financial Crisis)
   - Out-of-sample: 2019-2020 (COVID-19)
   - AUC improvement from 0.671 → 0.735 on unseen crisis
   - March 2020 crash specifically: elevated detection rates

3. **Integration > Isolation**
   - Novel interaction features rank highly in importance
   - Dispersion × Volatility captures systemic stress
   - EVT shape parameters contribute to tail detection
   - Combined approach outperforms single-method VaR

4. **Statistical Regime Change Detection**
   - Pre-crisis volatility: 14.8% annualized
   - Crisis volatility: 85.6% annualized (5.8× increase)
   - F-test p-value: < 0.0000001 (highly significant)
   - Correlations surge during crisis (diversification fails when needed most)

5. **Derivatives Pricing Implications**
   - Crisis GARCH volatility: 262% annualized (peak)
   - Option premiums: 10.5× higher than historical average
   - Demonstrates hedging cost explosion during regime shifts

### Practical Implications

The framework serves as a **risk committee screening tool** rather than automated trading signal:
- Early warning for tail risk regime shifts
- Model risk management (identifying VaR methodology failures)
- Stress testing and scenario analysis
- Dynamic hedge ratio adjustment
- Probability calibration needed before execution

---

## Data and Portfolio Construction

### Asset Selection

**Tickers**: JPM, GS, MS, C, BAC (Major U.S. Investment Banks)

**Rationale**: 
- Central to 2008 crisis (extreme stress test)
- Highly correlated during crisis (correlation breakdown study)
- Liquid markets (pricing reliability)
- Varying degrees of crisis impact

### Time Periods

| Period | Date Range | Purpose |
|--------|------------|---------|
| Pre-Crisis | Jan 2005 - Jun 2007 | Baseline "normal" regime |
| Crisis | Jul 2007 - Mar 2009 | Training on extreme events |
| Post-Crisis | Apr 2009 - Dec 2010 | Recovery dynamics |
| COVID-19 (OOS) | Jun 2019 - Dec 2020 | Out-of-sample validation |

### Portfolio Construction

```python
# Equal-weighted portfolio
weights = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
portfolio_returns = individual_returns @ weights
```

Equal weighting chosen for:
- Transparency and reproducibility
- No optimization bias
- Focus on systemic risk, not allocation

### Data Characteristics

- **Frequency**: Daily close-to-close returns
- **Total observations**: ~1,500 days (training period)
- **Missing data**: Yahoo Finance handles splits/dividends
- **Survivorship bias**: Lehman Brothers and Bear Stearns excluded (critical limitation discussed below)

---

## Methodology

### Exploratory Data Analysis

#### Return Distribution Characteristics

```python
def compute_return_statistics(returns_series):
    stats = {
        'Mean (Annualized)': returns_series.mean() * 252,
        'Std (Annualized)': returns_series.std() * np.sqrt(252),
        'Skewness': returns_series.skew(),
        'Excess Kurtosis': returns_series.kurtosis(),
        'Jarque-Bera': stats.jarque_bera(returns_series)
    }
    return stats
```

**Key Findings**:
- Crisis volatility 5.8× pre-crisis levels
- Excess kurtosis: 8.47 (vs 0 for normal distribution)
- Jarque-Bera strongly rejects normality (p < 0.001)
- Negative skewness during crisis (left tail heavier)

#### Structural Break Detection (Chow Test)

```python
def chow_test_volatility(returns_series, break_date):
    """F-test for variance equality across regimes"""
    pre_break = returns_series.loc[:break_date]
    post_break = returns_series.loc[break_date:]
    
    var_pre = pre_break.var()
    var_post = post_break.var()
    
    f_stat = var_post / var_pre  # if post > pre
    # Two-tailed test for variance ratio
    return f_stat, p_value
```

**Result**: Variance ratio of 33.5× with p-value < 10⁻¹⁰, confirming statistically significant regime change.

#### Dynamic Correlation Analysis

```python
def compute_rolling_correlations(returns_df, window=60):
    """Track correlation regime shifts"""
    avg_corrs = []
    for i in range(window, len(returns_df)):
        corr_matrix = returns_df.iloc[i-window:i].corr()
        upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
        avg_corrs.append(upper_tri.mean())
    return pd.Series(avg_corrs)
```

**Finding**: Average correlation increased from 0.58 (pre-crisis) to 0.81 (crisis)—a 40% increase. Diversification benefits evaporate precisely when needed most.

---

### GARCH Volatility Modeling

#### GARCH(1,1) Implementation

Custom maximum likelihood estimation from first principles:

```python
class GARCH:
    def __init__(self):
        self.params = None
        self.conditional_variance = None
    
    def _garch_variance(self, params, returns):
        omega, alpha, beta = params
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = returns.var()  # Initialize with sample variance
        
        for t in range(1, n):
            # Conditional variance equation
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        return sigma2
    
    def _neg_log_likelihood(self, params, returns):
        omega, alpha, beta = params
        # Stationarity constraint
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        sigma2 = self._garch_variance(params, returns)
        sigma2 = np.maximum(sigma2, 1e-10)  # Numerical stability
        
        # Gaussian log-likelihood
        ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(sigma2) + returns**2/sigma2)
        return -ll  # Minimize negative log-likelihood
    
    def fit(self, returns):
        initial_params = [1e-6, 0.1, 0.8]  # omega, alpha, beta
        bounds = [(1e-8, 1), (0, 1), (0, 1)]
        
        result = minimize(
            self._neg_log_likelihood, 
            initial_params, 
            args=(returns,),
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        self.params = result.x
        self.conditional_variance = self._garch_variance(self.params, returns)
        return self
```

**Model Interpretation**:
- σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
- α captures short-term shock impact (ARCH effect)
- β captures long-term persistence (GARCH effect)
- α + β < 1 ensures stationarity
- Typical result: α ≈ 0.09, β ≈ 0.89, persistence ≈ 0.98

#### GJR-GARCH (Leverage Effects)

Extended model capturing asymmetric volatility response:

```python
class GJR_GARCH:
    def _gjr_variance(self, params, returns):
        omega, alpha, gamma, beta = params
        sigma2 = np.zeros(n)
        
        for t in range(1, n):
            indicator = 1 if returns[t-1] < 0 else 0  # Negative shock
            sigma2[t] = omega + (alpha + gamma*indicator)*returns[t-1]**2 + beta*sigma2[t-1]
        return sigma2
```

**Leverage Effect**: Negative returns increase volatility more than positive returns of same magnitude—crucial for crisis modeling where downward spirals dominate.

---

### Extreme Value Theory

#### Threshold Selection (Mean Residual Life Plot)

```python
def mean_residual_life_plot(data, thresholds):
    """
    For GPD, mean excess should be linear in threshold.
    Point where linearity begins indicates appropriate threshold.
    """
    mean_excesses = []
    for u in thresholds:
        exceedances = data[data > u] - u
        if len(exceedances) > 10:
            mean_excesses.append(exceedances.mean())
    return thresholds, mean_excesses
```

**Selection Criterion**: 90th percentile of losses chosen as threshold, balancing:
- Sufficient exceedances for estimation (>50 observations)
- Focus on true tail behavior
- Mean residual life plot linearity

#### Generalized Pareto Distribution Fitting

```python
def fit_gpd(exceedances, threshold):
    """
    Fit GPD to exceedances over threshold.
    GPD is the limit distribution for exceedances (Pickands-Balkema-de Haan theorem).
    """
    excess = exceedances - threshold
    shape, loc, scale = genpareto.fit(excess, floc=0)  # loc=0 for exceedances
    return shape, scale
```

**Shape Parameter Interpretation**:
- ξ > 0: Heavy tails (Fréchet domain)—our case
- ξ = 0: Exponential tails (Gumbel domain)
- ξ < 0: Bounded tails (Weibull domain)

**Result**: ξ ≈ 0.15-0.25, confirming heavy-tailed distribution requiring EVT modeling.

#### EVT-Based VaR Estimation

```python
def evt_var(losses, threshold, xi, sigma, n_total, confidence=0.99):
    """
    VaR using POT-GPD method.
    More accurate for extreme quantiles than parametric methods.
    """
    n_exceed = np.sum(losses > threshold)
    exceed_prob = n_exceed / n_total
    
    if xi != 0:
        var = threshold + (sigma/xi) * (((1-confidence)/exceed_prob)**(-xi) - 1)
    else:
        var = threshold - sigma * np.log((1-confidence)/exceed_prob)
    return var
```

**Key Insight**: EVT-GPD VaR estimates are 35-50% higher than Gaussian VaR at 99% confidence, properly capturing fat tail risk.

---

### VaR Methodologies Comparison

#### Monte Carlo VaR with Cholesky Decomposition

```python
def monte_carlo_var_gbm(returns_df, n_simulations=10000, confidence=0.95):
    """
    Monte Carlo VaR preserving correlation structure.
    Uses Cholesky decomposition for correlated normal draws.
    """
    mu = returns_df.mean().values
    sigma = returns_df.std().values
    corr_matrix = returns_df.corr().values
    
    # Cholesky: Σ = LL^T, so L @ Z gives correlated normals
    L = np.linalg.cholesky(corr_matrix)
    
    simulated_losses = np.zeros(n_simulations)
    for i in range(n_simulations):
        Z = np.random.randn(n_assets)  # Independent normals
        correlated_Z = L @ Z  # Correlated normals
        asset_returns = mu + sigma * correlated_Z
        portfolio_return = weights @ asset_returns
        simulated_losses[i] = -portfolio_return
    
    var = np.percentile(simulated_losses, confidence * 100)
    cvar = simulated_losses[simulated_losses >= var].mean()  # Expected Shortfall
    return var, cvar
```

#### Methodology Comparison Results

| Method | 95% VaR | 99% VaR | Crisis Performance |
|--------|---------|---------|-------------------|
| Normal (Parametric) | 2.8% | 4.0% | Underestimates by 35% |
| Student-t | 3.2% | 5.1% | Partial capture |
| Historical Simulation | 3.1% | 4.8% | Backward-looking |
| Monte Carlo (GBM) | 2.9% | 4.2% | Assumes normality |
| GARCH | 3.5% | 5.8% | Dynamic, adapts |
| EVT-GPD | 4.1% | 7.2% | Best tail capture |

**Conclusion**: No single method dominates—integration is necessary.

---

### Machine Learning Pipeline

#### Walk-Forward Validation

```python
def walk_forward_validation(X, y, initial_train=504, step=63):
    """
    Expanding window cross-validation to avoid look-ahead bias.
    Mimics real-world deployment: train on past, predict future.
    
    - Initial training: 504 days (2 years)
    - Step size: 63 days (quarterly retraining)
    - Model refit at each step to capture evolving patterns
    """
    predictions, actuals = [], []
    train_end = initial_train
    
    while train_end + step <= len(X):
        # Train on all available history
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        
        # Test on next quarter
        X_test, y_test = X.iloc[train_end:train_end+step], y.iloc[train_end:train_end+step]
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,  # Prevent overfitting
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=scale_pos_weight  # Handle class imbalance
        )
        
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        predictions.extend(y_prob)
        actuals.extend(y_test.values)
        train_end += step
    
    return predictions, actuals, model
```

**Design Choices**:
- **XGBoost**: Handles non-linear interactions, built-in regularization
- **Max depth 3**: Interpretable trees, reduced overfitting
- **Quarterly retraining**: Balances adaptivity vs. stability
- **Scale pos weight**: Addresses ~5% tail event frequency

#### Performance Metrics

```python
# Walk-forward results
wf_auc = roc_auc_score(actuals, predictions)
print(f"AUC-ROC: {wf_auc:.4f}")  # 0.6714

# Precision-Recall at different thresholds
for thresh in [0.5, 0.3, 0.2, 0.1]:
    precision = precision_score(actuals, predictions >= thresh)
    recall = recall_score(actuals, predictions >= thresh)
```

**Interpretation**: AUC of 0.67 indicates moderate predictive power—better than random (0.5) but not highly accurate. Framework detects 67-74% of tail events at cost of false positives.

---

### Backtesting Framework

#### Kupiec Proportion of Failures Test

```python
def kupiec_pof_test(violations, n_total, expected_prob):
    """
    Tests if VaR violation frequency matches expected.
    H0: Actual violation rate = Expected rate (e.g., 5% for 95% VaR)
    
    Likelihood ratio test with chi-square(1) distribution.
    """
    n_violations = np.sum(violations)
    p = expected_prob
    x = n_violations
    n = n_total
    
    # LR statistic
    lr_stat = -2 * np.log(
        ((1-p)**(n-x) * p**x) / 
        ((1-x/n)**(n-x) * (x/n)**x)
    )
    
    p_value = 1 - chi2.cdf(lr_stat, 1)
    return lr_stat, p_value
```

**Purpose**: Unconditional coverage test—does VaR fail at the expected rate?

#### Christoffersen Independence Test

```python
def christoffersen_test(violations):
    """
    Tests if violations are independent (no clustering).
    Clustered violations indicate model misspecification.
    
    Transition matrix approach:
    - Count 0→0, 0→1, 1→0, 1→1 transitions
    - Test if P(violation|no prev violation) = P(violation|prev violation)
    """
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(violations)):
        if violations[i-1] == 0 and violations[i] == 0:
            n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1:
            n01 += 1
        # ... etc
    
    # Independence LR statistic
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n - 1)
    
    lr_ind = -2 * (log_likelihood_independence - log_likelihood_dependence)
    return lr_ind, p_value
```

**Purpose**: Conditional coverage test—are violations independent or clustered?

**Finding**: Pre-crisis VaR calibrated models fail both tests during crisis—violations are too frequent and clustered, indicating regime shift.

---

## Feature Engineering Architecture

The core innovation: transforming risk models into predictive features. The pipeline constructs 70+ features across six categories:

### Category 1: Traditional Rolling Statistics

```python
for w in [5, 10, 20, 40, 60]:
    features[f'volatility_{w}d'] = returns.rolling(w).std() * np.sqrt(252)
    features[f'skewness_{w}d'] = returns.rolling(w).skew()
    features[f'kurtosis_{w}d'] = returns.rolling(w).kurt()
    features[f'momentum_{w}d'] = returns.rolling(w).mean()
    features[f'min_return_{w}d'] = returns.rolling(w).min()
    features[f'range_{w}d'] = max - min
```

**Purpose**: Baseline features representing standard risk metrics.

### Category 2: GARCH-Based Dynamics (22% Total Importance)

```python
# 2a. Conditional volatility from fitted GARCH
features['garch_cond_vol'] = np.sqrt(garch_model.conditional_variance) * np.sqrt(252)

# 2b. Forecast error (realized - predicted) → regime shift indicator
realized_vol = returns.rolling(5).std() * np.sqrt(252)
features['garch_forecast_error'] = realized_vol - garch_cond_vol
features['garch_forecast_error_abs'] = np.abs(forecast_error)

# 2c. Volatility-of-volatility (second-order risk)
features['vol_of_vol_20d'] = garch_cond_vol.rolling(20).std()

# 2d. Volatility momentum (acceleration)
features['garch_vol_momentum_5d'] = garch_cond_vol.diff(5)

# 2e. Normalized volatility (z-score vs. recent history)
features['garch_vol_zscore'] = (garch_vol - garch_vol.rolling(60).mean()) / garch_vol.rolling(60).std()
```

**Key Insight**: GARCH conditional volatility provides 3× more predictive power than simple rolling volatility because it captures volatility clustering dynamics.

### Category 3: EVT Tail Parameters

```python
def rolling_evt_shape(returns, window=60, threshold_pct=90):
    """
    Track evolution of tail thickness over time.
    Increasing shape parameter (ξ) indicates fattening tails.
    """
    shape_params = []
    for i in range(len(returns)):
        if i < window:
            shape_params.append(np.nan)
        else:
            window_returns = returns.iloc[i-window:i]
            losses = -window_returns
            threshold = np.percentile(losses, threshold_pct)
            exceedances = losses[losses > threshold] - threshold
            
            if len(exceedances) >= 5:
                shape, _, scale = genpareto.fit(exceedances, floc=0)
                shape_params.append(shape)
    return shape_params

features['evt_shape_60d'] = rolling_evt_shape(returns, 60)
features['evt_shape_120d'] = rolling_evt_shape(returns, 120)
features['evt_shape_momentum'] = evt_shape_60d.diff(10)
features['evt_shape_zscore'] = (shape - mean) / std
```

**Purpose**: Dynamic tail risk monitoring—when ξ increases, tails are fattening and extreme events become more likely.

### Category 4: Cross-Sectional Stress Indicators

```python
# 4a. Return dispersion across portfolio
features['cross_sectional_dispersion'] = individual_returns.std(axis=1)

# 4b. Dynamic correlation (breakdown detection)
def rolling_avg_correlation(returns_df, window=20):
    for each window:
        corr_matrix = window_data.corr()
        upper_tri = extract_upper_triangle(corr_matrix)
        avg_corr = upper_tri.mean()
    return avg_corr

features['avg_correlation_20d'] = rolling_avg_correlation(individual_returns, 20)
features['correlation_change'] = avg_corr.diff(5)
```

**Systemic Risk Signal**: When individual stock returns diverge (high dispersion) AND correlations spike simultaneously, systemic stress is building.

### Category 5: Novel Interaction Features

```python
# 5a. GARCH vol × EVT shape (compound tail risk)
features['garch_evt_interaction'] = garch_cond_vol * evt_shape_60d

# 5b. Dispersion × Volatility (systemic stress signal) - #2 Feature Importance
features['dispersion_vol_interaction'] = cross_sectional_dispersion * garch_cond_vol

# 5c. Multi-indicator warning score
features['warning_signal_count'] = (
    (garch_vol_zscore > 1.5).astype(int) +
    (evt_shape_zscore > 1.5).astype(int) +
    (vol_ratio_5_20 > 1.2).astype(int) +
    (correlation_change > 0.05).astype(int)
)
```

**Novel Contribution**: The dispersion × volatility interaction ranks #2 in feature importance, capturing systemic stress dynamics that individual metrics miss.

### Category 6: Temporal Risk Patterns

```python
# Past exceedances (legitimate - happened before prediction)
threshold_95 = returns.quantile(0.05)
features['recent_exceedances_5d'] = (returns < threshold_95).rolling(5).sum()
features['recent_exceedances_20d'] = (returns < threshold_95).rolling(20).sum()
features['days_since_exceedance'] = cumulative_count_since_last_exceedance
```

**Purpose**: Tail events cluster—recent exceedances predict near-term exceedances.

---

## Model Validation

### SHAP Interpretability Analysis

```python
import shap
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)

# Feature importance decomposition
shap.summary_plot(shap_values, X_test)
```

**Top 10 Features by SHAP Importance**:

1. GARCH conditional volatility (22%)
2. Dispersion × Volatility interaction
3. 5-day rolling volatility
4. GARCH volatility momentum
5. EVT shape parameter (60d)
6. Cross-sectional dispersion
7. Volatility-of-volatility
8. Recent exceedances count
9. Average correlation (20d)
10. GARCH forecast error

**Validation**: SHAP confirms that GARCH/EVT-derived features dominate predictions over simple rolling statistics, validating the integration hypothesis.

### Out-of-Sample COVID-19 Validation

```python
# Train on 2005-2010 (Financial Crisis)
# Test on 2019-2020 (COVID-19) - COMPLETELY UNSEEN

covid_data = yf.download(['C', 'GS', 'JPM', 'MS'], start='2019-06-01', end='2020-12-31')
covid_features = create_advanced_ml_features(covid_returns, covid_individual, covid_garch)

# Apply trained model to new crisis
covid_predictions = final_model.predict_proba(covid_features)[:, 1]
covid_auc = roc_auc_score(covid_actuals, covid_predictions)

print(f"COVID-19 AUC: {covid_auc:.4f}")  # 0.735
print(f"2008 Crisis AUC: 0.6714")         # Training period
```

**Result**: Model performs BETTER on unseen crisis (0.735 vs 0.671), suggesting:
- Learned patterns are fundamental to crisis dynamics
- Not overfitting to 2008-specific idiosyncrasies  
- Framework captures transferable tail risk signatures

### Robustness Analysis

**Window Sensitivity Testing**:
- Varied rolling windows (5, 10, 20, 40, 60 days)
- Stable feature importance rankings
- GARCH features consistently dominate

**Threshold Sensitivity**:
- EVT threshold varied (85th-95th percentile)
- Results stable within reasonable range
- 90th percentile optimal for sample size

**Class Imbalance Handling**:
- Scale_pos_weight adjustment
- Precision-recall curves at multiple thresholds
- F1-score optimization for practical deployment

---

## Derivatives Pricing Extension

### Black-Scholes with GARCH Volatility

```python
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# Compare volatility regimes
vol_scenarios = {
    'Historical Average': 0.25,
    'Pre-Crisis GARCH': 0.15,
    'Crisis Peak GARCH': 2.62  # 262% annualized
}

# Results
# Historical: Call = $1.12
# Pre-Crisis: Call = $0.68
# Crisis Peak: Call = $11.76  (10.5× higher)
```

**Practical Implication**: Hedging costs explode during regime shifts. A risk committee using this framework would know that protective options become 10× more expensive precisely when protection is most valuable—informing capital allocation decisions.

### Expected Shortfall (CVaR) Analysis

Beyond VaR, the framework estimates Expected Shortfall—the average loss given that VaR is exceeded:

```python
def evt_expected_shortfall(threshold, xi, sigma, var_estimate):
    """Conditional tail expectation"""
    if xi < 1:
        es = (var_estimate + sigma - xi * threshold) / (1 - xi)
    else:
        es = np.inf  # Infinite expectation for heavy tails
    return es
```

**Result**: Crisis ES is 1.4× the VaR estimate, quantifying the severity of tail events beyond the threshold.

---

## Critical Limitations

### Survivorship Bias

**The Missing Data Problem**: Our analysis excludes Lehman Brothers and Bear Stearns—two institutions that *failed* during the crisis. Yahoo Finance delists bankrupt companies, so these extreme tail events are literally absent from our data.

**Impact**:
- Training on survivors only
- Most extreme losses (bankruptcy) excluded
- Systematic underestimation of true tail risk
- VaR calibrated to world where banks survive

**The Paradox**: The most informative data points for tail risk (actual failures) are the ones we can't study. This is analogous to analyzing plane crashes by interviewing only survivors.

**Implication**: All results should be interpreted as *conditional on survival*. True unconditional tail risk is likely more severe.

### Model Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|------------|
| **False Alarm Rate** | Model triggers warnings too frequently | Use as screening tool, not automated signal |
| **EVT Threshold Choice** | 90th percentile is arbitrary | Sensitivity analysis across thresholds |
| **Single Crisis Training** | Only 2008 data | COVID-19 validates some generalization |
| **Liquidity Not Modeled** | Assumes instant execution | Real hedges have transaction costs |
| **Probability Calibration** | Raw probabilities not actionable | Needs recalibration for trading |

### Statistical Limitations

- Limited tail event sample size (~75 events in training)
- Non-stationarity of financial markets
- Potential regime changes not captured by historical patterns
- Model complexity vs. interpretability tradeoff
- No transaction costs or market impact modeling

---

## Repository Structure

```
ml-integrated-tail-risk-framework/
├── VaR_Analysis_Advanced.ipynb    # Main analysis notebook (comprehensive)
├── README.md                      # This documentation
├── requirements.txt               # Python dependencies
├── data/                          # Data cache (if applicable)
│   └── .gitkeep
└── figures/                       # Generated visualizations
    └── .gitkeep
```

---

### Notebook Structure

The analysis is organized sequentially:

1. **Setup & Data Acquisition** (Cells 1-3)
   - Library imports
   - Data download via yfinance
   - Portfolio construction

2. **Exploratory Data Analysis** (Cells 4-12)
   - Return statistics by period
   - Structural break testing
   - Correlation regime analysis

3. **Risk Model Estimation** (Cells 13-18)
   - Monte Carlo VaR
   - GARCH/GJR-GARCH fitting
   - EVT threshold selection and GPD fitting

4. **Backtesting Framework** (Cells 19-21)
   - Kupiec test
   - Christoffersen test
   - Model validation

5. **Machine Learning Pipeline** (Cells 22-36)
   - Feature engineering
   - Walk-forward validation
   - SHAP interpretability
   - COVID-19 out-of-sample test

6. **Extensions & Conclusions** (Cells 37-47)
   - Derivatives pricing
   - Methodology comparison
   - Limitations discussion
   - Executive summary

---

## Future Work

### Methodological Extensions

1. **Copula-Based Tail Dependence**
   - Model non-linear dependence structure
   - Clayton/Gumbel copulas for asymmetric tails
   - Dynamic copula parameters

2. **Regime-Switching GARCH (MS-GARCH)**
   - Markov-switching volatility states
   - Sharper regime transition detection
   - Probability of regime membership

3. **Alternative Data Integration**
   - VIX term structure (forward-looking)
   - Credit default swap spreads
   - Options-implied volatility surface
   - News sentiment indicators

4. **Advanced ML Architectures**
   - LSTM for temporal dependencies
   - Attention mechanisms for crisis detection
   - Ensemble methods (stacking)
   - Uncertainty quantification (probabilistic predictions)

5. **Network-Based Systemic Risk**
   - Granger causality networks
   - Contagion modeling
   - Centrality measures as features

### Practical Enhancements

6. **Probability Calibration**
   - Platt scaling
   - Isotonic regression
   - Actionable hedging signals

7. **Transaction Cost Integration**
   - Market impact modeling
   - Optimal execution
   - Net profit after costs

8. **Real-Time Deployment**
   - Feature computation optimization
   - Streaming data pipeline
   - Alert system integration

9. **Multi-Asset Validation**
   - Credit markets
   - Commodities
   - Foreign exchange
   - Cross-asset contagion

10. **Regulatory Compliance**
    - Basel III/IV alignment
    - Stress testing integration
    - Model risk management framework

---


## Technical Appendix

### Mathematical Foundations

**GARCH(1,1) Log-Likelihood**:

$$\mathcal{L}(\theta) = -\frac{1}{2}\sum_{t=1}^{T}\left[\log(2\pi) + \log(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2}\right]$$

**GPD Distribution**:

$$F(x) = 1 - \left(1 + \frac{\xi x}{\sigma}\right)^{-1/\xi}, \quad x > 0$$

**POT-VaR Formula**:

$$\text{VaR}_\alpha = u + \frac{\sigma}{\xi}\left[\left(\frac{1-\alpha}{F(u)}\right)^{-\xi} - 1\right]$$

**Christoffersen LR Statistic**:

$$LR_{ind} = -2\log\left[\frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}{(1-\pi_{01})^{n_{00}}\pi_{01}^{n_{01}}(1-\pi_{11})^{n_{10}}\pi_{11}^{n_{11}}}\right] \sim \chi^2(1)$$

---

## Acknowledgments

This project was developed as independent research into quantitative risk management methodologies. The work combines classical financial econometrics with modern machine learning techniques, emphasizing interpretability and practical applicability for risk management contexts.

The custom implementations of GARCH, EVT, and backtesting frameworks prioritize pedagogical clarity and understanding of first principles over production optimization.


