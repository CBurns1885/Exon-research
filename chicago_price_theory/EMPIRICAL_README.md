# Empirical Data Integration - Chicago Price Theory Models

This extension adds **real-world economic data** capabilities to all theoretical models while maintaining full backward compatibility.

## Two Modes of Operation

### 1. Theoretical Mode (Default)
**No API keys needed** - specify all parameters manually

```python
from chicago_price_theory import LaborMarketModel

model = LaborMarketModel(wage=25.0, time_endowment=16.0)
result = model.labor_supply_individual()
```

### 2. Empirical Mode (Optional)
**Load real economic data** from APIs

```python
from chicago_price_theory.empirical_models import EmpiricalLaborMarketModel

model = EmpiricalLaborMarketModel()
data = model.load_from_fred(api_key='your_key')  # Uses real wage data
result = model.labor_supply_individual()
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Free API Keys

#### FRED (Federal Reserve Economic Data) - **Recommended**
- Most comprehensive US economic data
- **Free, unlimited access**
- Get key: https://fred.stlouisfed.org/docs/api/api_key.html
- ~1 minute registration

#### BLS (Bureau of Labor Statistics) - Optional
- Employment and wage data
- Free tier: 25 queries/day (usually sufficient)
- Get key: https://data.bls.gov/registrationEngine/

#### World Bank - No key required
- International development data
- Works out of the box

### 3. Configure API Keys

**Option A: Environment Variable** (Recommended)
```bash
export FRED_API_KEY='your_api_key_here'
export BLS_API_KEY='your_bls_key_optional'
```

**Option B: Config File**
```bash
# Copy template
cp chicago_price_theory/config.json.template chicago_price_theory/config.json

# Edit config.json and add your keys
{
  "fred_api_key": "your_actual_key_here",
  "bls_api_key": "optional"
}
```

---

## Available Data Sources

### Labor Market Data
```python
from chicago_price_theory.empirical_models import EmpiricalLaborMarketModel

model = EmpiricalLaborMarketModel()

# Load real wage and unemployment data
data = model.load_from_fred(
    wage_series='CES0500000003',      # Avg hourly earnings
    unemployment_series='UNRATE',      # Unemployment rate
    api_key='your_key'
)

print(f"Current avg wage: ${data['wage']:.2f}/hour")
print(f"Unemployment: {data['unemployment_rate']:.1f}%")

# Model now uses real data
eq = model.market_equilibrium()
```

**Available FRED Series:**
- `CES0500000003` - Average Hourly Earnings (Total Private)
- `UNRATE` - Unemployment Rate
- `CIVPART` - Labor Force Participation Rate
- `PAYEMS` - Total Nonfarm Payroll Employment

### Human Capital / Education Data
```python
from chicago_price_theory.empirical_models import EmpiricalHumanCapitalModel

model = EmpiricalHumanCapitalModel()

# Load real wage-education data
data = model.load_education_wage_data()

print(f"High school wage: ${data['high_school_wage']:,.0f}/year")
print(f"College wage: ${data['college_wage']:,.0f}/year")
print(f"College premium: {data['college_premium']:.1f}%")

# Or estimate Mincer equation from micro data
results = model.estimate_mincer_equation(
    education_data=years_of_schooling,
    experience_data=years_of_experience,
    wage_data=hourly_wages
)

print(f"Return to education: {results['return_to_education']*100:.1f}% per year")
```

### Commodity Price Data
```python
from chicago_price_theory.empirical_models import EmpiricalSupplyDemandModel

model = EmpiricalSupplyDemandModel()

# Load real commodity prices
data = model.load_commodity_data('oil', api_key='your_key')

print(f"Current oil price: ${data['latest_price']:.2f}/barrel")

# Available commodities:
# - 'oil' (Crude Oil WTI)
# - 'wheat' (Wheat prices)
# - 'gold' (Gold prices)
# - 'gasoline' (Retail gasoline)
# - 'housing' (Case-Shiller Home Price Index)
```

### General Economic Indicators
```python
from chicago_price_theory.data_sources import get_economic_data

# Quick data fetch
df = get_economic_data('unemployment', source='fred', api_key='your_key')
print(df.tail())

# Available indicators:
# - 'unemployment'
# - 'wages'
# - 'cpi' (inflation)
# - 'gdp'
# - 'employment'
```

---

## Model Calibration

### Estimate Parameters from Real Data

```python
from chicago_price_theory.calibration import (
    SupplyDemandCalibrator,
    LaborMarketCalibrator,
    HumanCapitalCalibrator
)

# Example 1: Calibrate demand curve
calibrator = SupplyDemandCalibrator()
a, b = calibrator.calibrate_linear(
    price_data=observed_prices,
    quantity_data=observed_quantities,
    curve_type='demand'
)
print(f"Estimated demand: Q = {a:.2f} - {b:.2f}*P")

# Example 2: Estimate labor supply elasticity
labor_cal = LaborMarketCalibrator()
elasticity = labor_cal.estimate_labor_supply_elasticity(
    wage_data=wages,
    hours_data=hours_worked
)
print(f"Labor supply elasticity: {elasticity:.2f}")

# Example 3: Mincer equation (returns to education)
hc_cal = HumanCapitalCalibrator()
results = hc_cal.estimate_mincer_equation(
    education_years=schooling,
    experience_years=experience,
    wages=wages
)
print(f"Return to education: {results['return_to_education']*100:.1f}% per year")
```

---

## Complete Examples

### Example 1: Labor Market with Real Data

```python
from chicago_price_theory.empirical_models import EmpiricalLaborMarketModel

# Create model
model = EmpiricalLaborMarketModel()

# Load real wage data from FRED
data = model.load_from_fred(api_key='your_fred_key')

print(f"Current average wage: ${data['wage']:.2f}/hour")
print(f"Current unemployment: {data['unemployment_rate']:.1f}%")

# Run analysis with real data
individual = model.labor_supply_individual()
print(f"Optimal hours worked: {individual['hours_worked']:.2f}")

# Market equilibrium
eq = model.market_equilibrium(n_workers=100, n_firms=10)
print(f"Equilibrium wage: ${eq['wage']:.2f}/hour")

# Policy analysis: minimum wage effect
mw = model.minimum_wage_effect(
    min_wage=data['wage'] * 1.25,  # 25% above current
    n_workers=100,
    n_firms=10
)
print(f"Unemployment created: {mw['unemployment_rate']:.1f}%")
```

### Example 2: Education Returns from Real Data

```python
from chicago_price_theory.empirical_models import EmpiricalHumanCapitalModel

model = EmpiricalHumanCapitalModel()

# Load real education-wage data
data = model.load_education_wage_data()

print("Median Weekly Earnings by Education:")
for level, wage in data['wage_by_education'].items():
    print(f"  {level}: ${wage:,.0f}/week")

print(f"\nCollege Premium: {data['college_premium']:.1f}%")

# Model is now calibrated with real return to education
opt = model.optimal_schooling()
print(f"\nOptimal years of schooling: {opt['optimal_years']:.1f}")
print(f"NPV: ${opt['npv']:,.0f}")
print(f"IRR: {opt['internal_rate_of_return']*100:.1f}%")
```

### Example 3: Estimate Mincer Equation from Micro Data

```python
import numpy as np
from chicago_price_theory.empirical_models import EmpiricalHumanCapitalModel

# Load individual-level data (e.g., from CPS, PSID)
# Here using simulated data
education = np.array([...])  # Years of schooling
experience = np.array([...])  # Years of experience
wages = np.array([...])       # Hourly wages

# Estimate Mincer equation
model = EmpiricalHumanCapitalModel()
results = model.estimate_mincer_equation(education, experience, wages)

print("Estimated Mincer Equation:")
print(f"  log(wage) = {results['intercept']:.3f}")
print(f"              + {results['return_to_education']:.3f} * education")
print(f"              + {results['experience_coef']:.3f} * experience")
print(f"              + {results['experience_sq_coef']:.6f} * experience²")
print(f"\n  R-squared: {results['r_squared']:.3f}")

print(f"\nInterpretation:")
print(f"Each year of education → {results['return_to_education']*100:.1f}% wage increase")
```

---

## Data Sources Reference

### FRED (Federal Reserve Economic Data)

**Most Popular Series:**

| Series ID | Description |
|-----------|-------------|
| `UNRATE` | Unemployment Rate (%) |
| `CES0500000003` | Avg Hourly Earnings - Total Private |
| `CPIAUCSL` | Consumer Price Index |
| `GDP` | Gross Domestic Product |
| `PAYEMS` | Total Nonfarm Payroll Employment |
| `CIVPART` | Labor Force Participation Rate |
| `DCOILWTICO` | Crude Oil Prices: WTI |
| `GASREGW` | Gasoline Prices (Regular) |
| `CSUSHPISA` | Case-Shiller Home Price Index |

Browse all series: https://fred.stlouisfed.org/

### BLS (Bureau of Labor Statistics)

**Key Series:**

| Series ID | Description |
|-----------|-------------|
| `LNS14000000` | Unemployment Rate |
| `CES0500000003` | Avg Hourly Earnings |
| `CUUR0000SA0` | Consumer Price Index - All Urban Consumers |

Browse: https://www.bls.gov/data/

### World Bank

**Popular Indicators:**

| Indicator | Description |
|-----------|-------------|
| `NY.GDP.PCAP.CD` | GDP per capita |
| `SE.ADT.LITR.ZS` | Literacy rate |
| `SL.UEM.TOTL.ZS` | Unemployment rate |
| `SP.POP.TOTL` | Total population |

Browse: https://data.worldbank.org/

---

## Error Handling

All data fetching is designed to fail gracefully:

```python
try:
    data = model.load_from_fred(api_key='your_key')
except ValueError as e:
    print(f"API key issue: {e}")
except ConnectionError as e:
    print(f"Network issue: {e}")
except Exception as e:
    print(f"Other error: {e}")
    # Fall back to theoretical mode
    model = LaborMarketModel(wage=25.0)  # Manual parameters
```

---

## Best Practices

### 1. Start with Theoretical Mode
```python
# First, understand the model with simple parameters
model = LaborMarketModel(wage=25.0)
result = model.labor_supply_individual()
```

### 2. Then Add Real Data
```python
# Once comfortable, enhance with real data
from empirical_models import EmpiricalLaborMarketModel
model = EmpiricalLaborMarketModel()
data = model.load_from_fred(api_key='...')
```

### 3. Validate Results
```python
# Compare theoretical predictions to empirical calibrations
theoretical = LaborMarketModel(wage=20.0)
empirical = EmpiricalLaborMarketModel()
empirical.load_from_fred(api_key='...')

# Check if predictions are similar
print(f"Theoretical: {theoretical.labor_supply_individual()['hours_worked']:.2f}")
print(f"Empirical: {empirical.labor_supply_individual()['hours_worked']:.2f}")
```

---

## Limitations & Caveats

### Data Limitations
- **Identification Problems**: Estimating supply and demand simultaneously requires instruments
- **Omitted Variables**: Simple regressions may have bias (e.g., ability bias in education returns)
- **Sample Selection**: Non-random samples can bias estimates

### Chicago Methodology
- Use **instrumental variables** when endogeneity is a concern
- Conduct **robustness checks** with different specifications
- Test **theoretical predictions** against calibrated models
- Consider **natural experiments** for causal identification

### Example: Addressing Ability Bias
```python
# Naive estimate (biased upward)
results = model.estimate_mincer_equation(education, experience, wages)
print(f"Naive return: {results['return_to_education']*100:.1f}%")

# Chicago approach: Use IV or natural experiments
# - Twin studies (genetic factors constant)
# - Compulsory schooling laws (policy variation)
# - Geographic variation in college access
```

---

## Troubleshooting

### "API key required"
Get free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html

### "Module not found"
```bash
pip install -r requirements.txt
```

### "No data found for series"
- Check series ID is correct
- Check date range (some series have limited history)
- Try different date range

### Rate Limits
- FRED: No rate limits with API key
- BLS: 25 queries/day (free), 500/day (with registration)
- Solution: Cache data locally for repeated analysis

---

## Advanced: Custom Data Sources

```python
from chicago_price_theory.data_sources import FREDDataFetcher

# Fetch any FRED series
fred = FREDDataFetcher(api_key='your_key')
df = fred.fetch_series(
    series_id='CUSTOMSERIES',
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# Use in your analysis
prices = df['value'].values
# ... calibrate model
```

---

## Support & Resources

- **FRED Documentation**: https://fred.stlouisfed.org/docs/api/
- **BLS API Guide**: https://www.bls.gov/developers/
- **Chicago Price Theory**: See main README.md
- **Issues**: GitHub Issues for bug reports

---

## Summary

✅ **Theoretical models**: Work without any API keys
✅ **Empirical enhancement**: Optional real-data integration
✅ **Easy setup**: Free API keys, 5-minute setup
✅ **Flexible**: Use theory alone or combine with data
✅ **Educational**: Learn theory, then confront with reality

**The Chicago Way**: Theory guides empirical work, data tests theory.
