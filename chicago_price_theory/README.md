# Chicago Price Theory: Economic Models Collection

A comprehensive collection of economic models implementing core principles of Chicago School price theory, emphasizing rational choice, price mechanisms, and market equilibrium.

## Overview of Chicago Price Theory

Chicago price theory, developed by economists at the University of Chicago (Milton Friedman, George Stigler, Gary Becker, and others), focuses on:

1. **Price as the Central Coordinating Mechanism**: Prices transmit information and coordinate economic activity
2. **Rational Optimization**: Agents maximize utility/profit subject to constraints
3. **Marginal Analysis**: Decisions made at the margin
4. **Empirical Testing**: Theory must be testable and confronted with data
5. **Broad Application**: Economic reasoning applies to all human behavior

## Models Included

### 1. Supply and Demand Equilibrium
- Classic market clearing model
- Price adjustment dynamics (tatonnement process)
- Comparative statics analysis
- Elasticity calculations

### 2. Consumer Theory
- Utility maximization with budget constraints
- Indifference curve analysis
- Income and substitution effects
- Engel curves and demand derivation

### 3. Producer Theory
- Profit maximization
- Cost minimization
- Short-run vs long-run analysis
- Supply curve derivation

### 4. Labor Market Model
- Labor supply (leisure-income tradeoff)
- Labor demand (marginal productivity theory)
- Wage determination
- Employment effects of policy interventions

### 5. Human Capital Investment (Becker)
- Education as investment
- Present value calculations
- Optimal schooling decisions
- Returns to education

### 6. Price Discrimination
- First-degree (perfect price discrimination)
- Second-degree (quantity discounts)
- Third-degree (market segmentation)
- Welfare analysis

### 7. Search Theory
- Optimal job search with costs
- Reservation wage determination
- Expected duration of search
- Value of information

### 8. Time Allocation (Becker)
- Household production function
- Time as a constraint
- Full income concept
- Allocation between market and home production

## Usage

```python
from chicago_price_theory import SupplyDemandModel, ConsumerModel, LaborMarketModel

# Example: Supply and Demand
model = SupplyDemandModel()
equilibrium = model.find_equilibrium()
model.plot_market()

# Example: Consumer Choice
consumer = ConsumerModel(income=1000, prices=[10, 20])
optimal_bundle = consumer.maximize_utility()
consumer.plot_indifference_curves()
```

## Theoretical Foundations

Each model is built on core Chicago principles:
- **No Free Lunch**: Every choice involves opportunity costs
- **People Respond to Incentives**: Price changes affect behavior
- **Markets Clear**: Tendency toward equilibrium through price adjustment
- **Substitution is Ubiquitous**: Agents substitute at the margin

## Installation

```bash
pip install numpy scipy matplotlib
```

## Running the Models

```bash
python -m chicago_price_theory.main
```

This will run interactive demonstrations of all models with visualizations.
