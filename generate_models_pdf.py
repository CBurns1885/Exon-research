"""Generate Exon Research Models Reference PDF using ReportLab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)

WIDTH, HEIGHT = A4
MARGIN = 2 * cm


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=32, leading=38, textColor=HexColor("#14397A"),
        alignment=TA_CENTER, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "CoverSub", parent=styles["Normal"],
        fontSize=16, leading=20, textColor=HexColor("#555555"),
        alignment=TA_CENTER, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "SectionTitle", parent=styles["Heading1"],
        fontSize=18, leading=22, textColor=HexColor("#14397A"),
        spaceBefore=20, spaceAfter=8,
        borderWidth=1, borderColor=HexColor("#14397A"), borderPadding=4,
    ))
    styles.add(ParagraphStyle(
        "ModelTitle", parent=styles["Heading2"],
        fontSize=13, leading=16, textColor=HexColor("#28508C"),
        spaceBefore=14, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=13, textColor=HexColor("#1E1E1E"),
        alignment=TA_JUSTIFY, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "Formula", parent=styles["Code"],
        fontSize=8.5, leading=11, textColor=HexColor("#500000"),
        backColor=HexColor("#F5F5F5"), borderWidth=0.5,
        borderColor=HexColor("#CCCCCC"), borderPadding=6,
        spaceAfter=6, spaceBefore=4, leftIndent=10, rightIndent=10,
    ))
    styles.add(ParagraphStyle(
        "ProLabel", parent=styles["Normal"],
        fontSize=10, leading=12, textColor=HexColor("#007800"),
        fontName="Helvetica-Bold", spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "ConLabel", parent=styles["Normal"],
        fontSize=10, leading=12, textColor=HexColor("#B40000"),
        fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#1E1E1E"),
        leftIndent=15, bulletIndent=5,
    ))
    styles.add(ParagraphStyle(
        "Example", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=HexColor("#3C3C3C"),
        backColor=HexColor("#F0F8FF"), borderWidth=0.5,
        borderColor=HexColor("#B0D0F0"), borderPadding=6,
        spaceAfter=10, spaceBefore=4, fontName="Helvetica-Oblique",
        leftIndent=10, rightIndent=10,
    ))
    styles.add(ParagraphStyle(
        "TOCItem", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#1E1E1E"),
    ))
    styles.add(ParagraphStyle(
        "TOCIndent", parent=styles["Normal"],
        fontSize=10, leading=14, textColor=HexColor("#444444"),
        leftIndent=20,
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"],
        fontSize=7, leading=9, textColor=HexColor("#1E1E1E"),
    ))
    return styles


def add_model(story, s, title, description, formula, pros, cons, example):
    """Add a full model section."""
    elements = []
    elements.append(Paragraph(title, s["ModelTitle"]))
    elements.append(Paragraph(description, s["Body"]))
    if formula:
        elements.append(Paragraph(formula.replace("\n", "<br/>"), s["Formula"]))
    elements.append(Paragraph("Strengths:", s["ProLabel"]))
    for p in pros:
        elements.append(Paragraph(f"• {p}", s["BulletItem"]))
    elements.append(Paragraph("Weaknesses:", s["ConLabel"]))
    for c in cons:
        elements.append(Paragraph(f"• {c}", s["BulletItem"]))
    elements.append(Paragraph(f"<b>Example:</b> {example}", s["Example"]))
    story.append(KeepTogether(elements))


def build_pdf():
    doc = SimpleDocTemplate(
        "/home/user/Exon-research/Exon_Research_Models_Reference.pdf",
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    s = get_styles()
    story = []

    # ── COVER ──
    story.append(Spacer(1, 80))
    story.append(Paragraph("Exon Research", s["CoverTitle"]))
    story.append(Paragraph("Quantitative Models Reference Guide", s["CoverSub"]))
    story.append(Spacer(1, 15))
    story.append(Paragraph("Mathematical Models, Statistical Methods &amp; Machine Learning<br/>for Crypto, Equity &amp; Options Trading", s["CoverSub"]))
    story.append(Spacer(1, 30))
    story.append(Paragraph("31 quantitative models across 6 asset-class modules", s["CoverSub"]))
    story.append(Paragraph("260 automated tests | Walk-forward validated", s["CoverSub"]))
    story.append(Spacer(1, 40))
    story.append(Paragraph("June 2026", s["CoverSub"]))
    story.append(PageBreak())

    # ── TABLE OF CONTENTS ──
    story.append(Paragraph("Table of Contents", s["SectionTitle"]))
    toc_sections = [
        "1. Options Pricing Models",
        "2. Momentum &amp; Trend Models",
        "3. Mean-Reversion Models",
        "4. Statistical &amp; Signal Processing Models",
        "5. Regime Detection &amp; Factor Models",
        "6. Portfolio Optimisation Models",
        "7. Risk Management Models",
        "8. Backtesting &amp; Validation",
        "9. Machine Learning Models",
        "10. IV Surface &amp; Volatility Models",
    ]
    toc_items = {
        "1. Options Pricing Models": ["1.1 Black-Scholes Model", "1.2 Greeks (Delta, Gamma, Theta, Vega, Rho)", "1.3 Implied Volatility (Brent's Method)", "1.4 Put-Call Parity"],
        "2. Momentum &amp; Trend Models": ["2.1 Time-Series Momentum", "2.2 Cross-Sectional Momentum", "2.3 Adaptive Momentum", "2.4 Sector Rotation"],
        "3. Mean-Reversion Models": ["3.1 Bollinger Band Mean Reversion", "3.2 Ornstein-Uhlenbeck Process", "3.3 Pairs Trading (Engle-Granger)"],
        "4. Statistical &amp; Signal Processing Models": ["4.1 Kalman Filter", "4.2 Discrete Wavelet Transform", "4.3 Hurst Exponent", "4.4 Principal Component Analysis"],
        "5. Regime Detection &amp; Factor Models": ["5.1 Gaussian Mixture Model", "5.2 Multi-Factor Model (IC-Weighted)", "5.3 PEAD", "5.4 Market Microstructure"],
        "6. Portfolio Optimisation Models": ["6.1 Mean-Variance (Markowitz)", "6.2 Risk Parity", "6.3 Kelly Criterion"],
        "7. Risk Management Models": ["7.1 VaR &amp; Expected Shortfall", "7.2 Volatility Targeting", "7.3 Drawdown Circuit Breaker"],
        "8. Backtesting &amp; Validation": ["8.1 Walk-Forward Analysis", "8.2 Monte Carlo Sharpe Test", "8.3 Options Backtesting Engine"],
        "9. Machine Learning Models": ["9.1 XGBoost", "9.2 LightGBM"],
        "10. IV Surface &amp; Volatility Models": ["10.1 IV Rank &amp; IV Percentile", "10.2 Variance Risk Premium", "10.3 Signal-to-Options Structure Mapping"],
    }
    for sec in toc_sections:
        story.append(Paragraph(f"<b>{sec}</b>", s["TOCItem"]))
        for item in toc_items.get(sec, []):
            story.append(Paragraph(item, s["TOCIndent"]))
    story.append(PageBreak())

    # ── 1. OPTIONS PRICING ──
    story.append(Paragraph("1. Options Pricing Models", s["SectionTitle"]))

    add_model(story, s,
        "1.1  Black-Scholes Model",
        "The foundational options pricing model derived by Fischer Black and Myron Scholes (1973). It prices European-style options by modelling the underlying as geometric Brownian motion under risk-neutral valuation. In Exon, this is the core engine for options backtesting (mark-to-market repricing every bar) and for estimating theoretical premiums when live market data is unavailable.",
        "Call = S * N(d1) - K * exp(-rT) * N(d2)\nPut  = K * exp(-rT) * N(-d2) - S * N(-d1)\n\nd1 = [ln(S/K) + (r + 0.5*v^2)*T] / (v*sqrt(T))\nd2 = d1 - v*sqrt(T)\n\nS=spot, K=strike, T=time to expiry (years), v=volatility, r=risk-free rate",
        ["Closed-form solution: instant computation, no iteration needed",
         "Mathematically elegant with well-understood properties",
         "Foundation for all Greeks calculations",
         "Widely accepted as market standard for European options"],
        ["Assumes constant volatility (reality: volatility smile/skew)",
         "Assumes log-normal returns (reality: fat tails, jumps)",
         "European-style only (no early exercise modelling)",
         "Poor for deep OTM/ITM options and near expiry"],
        "AAPL at $150, strike $155, 30 days to expiry, 25% IV, 5% rate: Call price = $2.47 (mostly time value since OTM). If AAPL moves to $160, BS reprices the call to ~$7.30, capturing the delta-driven profit."
    )

    add_model(story, s,
        "1.2  Greeks (Delta, Gamma, Theta, Vega, Rho)",
        "The partial derivatives of the Black-Scholes price with respect to underlying parameters. Greeks quantify how option price changes when each input moves. In Exon, portfolio-level Greeks are tracked daily during backtesting to monitor risk exposure.",
        "Delta = dV/dS      (N(d1) for calls, N(d1)-1 for puts)\nGamma = d2V/dS2    (pdf(d1) / (S*v*sqrt(T)))\nTheta = dV/dT      (time decay per day)\nVega  = dV/dv      (per 1% IV change)\nRho   = dV/dr      (per 1% rate change)",
        ["Complete sensitivity profile for any position",
         "Enables hedging: delta-neutral portfolios, gamma scalping",
         "Theta quantifies time decay cost (or income for sellers)",
         "Vega captures volatility exposure precisely"],
        ["Point-in-time estimates: change as underlying moves",
         "Higher-order Greeks (charm, vanna, volga) not computed",
         "Assumes continuous hedging (discrete rebalancing in practice)",
         "Gamma risk amplifies near expiry and at-the-money"],
        "An iron condor on SPY: net delta ~0 (market neutral), net theta +$15/day (time decay income), net vega -$200 (short volatility). If VIX spikes 5%, vega tells you the position loses ~$1000."
    )

    add_model(story, s,
        "1.3  Implied Volatility Solver (Brent's Method)",
        "Implied volatility is the market's consensus forecast of future volatility, extracted by inverting the Black-Scholes formula. Since BS has no closed-form inverse for vol, Exon uses Brent's root-finding method (bracketed between 0.1% and 1000% IV) to find the volatility that makes the BS price equal the market price.",
        "Find v* such that: BS(S, K, T, v*, r) = Market_Price\n\nBrent's method: hybrid bisection + secant + inverse quadratic\ninterpolation. Guaranteed convergence on [0.001, 10.0].\nTolerance: 1e-6",
        ["Guaranteed convergence (unlike Newton-Raphson which can diverge)",
         "Super-linear convergence speed in practice",
         "Robust across all strike/expiry combinations"],
        ["Slower than Newton-Raphson when derivatives are cheap",
         "Requires valid bracket [a,b] where f(a)*f(b) < 0",
         "Returns NaN for invalid inputs (arbitrage-violating prices)"],
        "MSFT call trading at $8.50, spot $420, strike $425, 45 DTE: Brent solver finds IV = 22.3%. Compare to 30-day realized vol of 18.1% to identify that options are 'expensive' (IV > RV), favouring premium-selling strategies."
    )

    add_model(story, s,
        "1.4  Put-Call Parity",
        "A fundamental arbitrage relationship between European call and put prices. Exon uses this as a sanity check: if parity is violated beyond tolerance, something is wrong with the data or pricing.",
        "C - P = S - K * exp(-rT)\n\nDeviation = (C - P) - (S - K*exp(-rT))\nParity holds if |deviation| <= tolerance * S",
        ["Model-free: works regardless of volatility assumptions",
         "Detects data errors and stale quotes",
         "Confirms market efficiency in real-time"],
        ["Only applies to European options (American options have early exercise premium)",
         "Dividend adjustments needed for equity options",
         "Bid-ask spreads can cause apparent violations in practice"],
        "SPY call at $12.50, put at $8.30, spot $580, strike $575, 30 DTE, r=5%: Theoretical diff = $7.35. Actual diff = $4.20. Deviation = -$3.15 flags a data issue."
    )

    story.append(PageBreak())

    # ── 2. MOMENTUM ──
    story.append(Paragraph("2. Momentum &amp; Trend Models", s["SectionTitle"]))

    add_model(story, s,
        "2.1  Time-Series Momentum (TSMOM)",
        "Based on Moskowitz, Ooi &amp; Pedersen (2012). Measures an asset's own historical return over multiple lookback windows and trades in the direction of the trend. Each window's signal is z-score normalised and combined. Position is volatility-scaled to target a fixed annualised volatility level.",
        "Signal_w = (Price_t / Price_{t-w} - 1)  for each window w\nZ_w = (Signal_w - rolling_mean) / rolling_std\nCombined = mean(Z_w for all windows)\nVol_scale = target_vol / realised_vol\nFinal_position = Combined * Vol_scale\n\nDefault lookbacks: [24, 72, 168, 336, 720] hours or [5, 21, 63, 126, 252] days",
        ["Academically robust (published factor with decades of evidence)",
         "Multi-window approach reduces whipsaws from any single lookback",
         "Volatility scaling equalises risk across assets",
         "Works across all asset classes"],
        ["Suffers in choppy/range-bound markets (trend reversals)",
         "Lagging indicator: enters after move has started",
         "Crowded trade: many quant funds use momentum",
         "Volatility scaling can amplify losses in regime changes"],
        "BTC 720-hour return = +35%, z-score = 2.1 (strong uptrend). 72-hour return = +8%, z-score = 1.4. Combined = 0.875. Realised vol = 60%, target = 15%, scale = 0.25. Final position: long 22%."
    )

    add_model(story, s,
        "2.2  Cross-Sectional Momentum (XSMOM)",
        "Ranks all assets by their recent return performance and goes long the winners while shorting the losers. Based on Jegadeesh &amp; Titman (1993). Unlike TSMOM which trades an asset against itself, this trades assets against each other.",
        "Return_i = Price_i(t) / Price_i(t-lookback) - 1\nRank assets by Return_i\nLong top_n, Short bottom_n\nStrength = 1 - rank/n * 0.5\n\nDefault: lookback=63 days, top_n=5, bottom_n=3",
        ["Market-neutral: long-short cancels market beta",
         "Captures relative value (best vs worst performers)",
         "Robust across global equities for 200+ years of data"],
        ["Momentum crashes: sharp reversals destroy short-term gains",
         "High turnover: frequent rebalancing increases costs",
         "Works best with large universes (>20 assets)"],
        "15-stock universe: NVDA +28% (rank 1), AAPL +15% (rank 2), ... INTC -12% (rank 14). Long NVDA/AAPL/GOOGL, short INTC/WBA. Winner strength = 1.0, loser = -0.8."
    )

    add_model(story, s,
        "2.3  Adaptive Momentum (Efficiency Ratio)",
        "Combines Kaufman's Efficiency Ratio with volatility regime detection to dynamically adjust the momentum lookback window. In trending markets, uses longer lookbacks; in choppy markets, shortens to reduce whipsaw losses.",
        "Efficiency Ratio = |Price_t - Price_{t-n}| / sum(|Price_i - Price_{i-1}|)\n\nER -> 1.0: strong trend (price moved directly)\nER -> 0.0: choppy (lots of back-and-forth)\n\nAdaptive_lookback = base * (2.0 - ER)\nConfidence boosted when ER > 0.5, penalised when < 0.3",
        ["Adapts to market conditions automatically",
         "Reduces whipsaws in choppy markets",
         "No regime detection model needed (self-adjusting)"],
        ["More parameters to tune than simple momentum",
         "Lag in adapting to sudden regime shifts",
         "Harder to backtest: path-dependent lookback changes"],
        "SPY with ER=0.72 (strong uptrend): lookback shortens, confidence boosted 1.3x. Next month ER drops to 0.15 (choppy): lookback extends, confidence cut to 0.5x."
    )

    add_model(story, s,
        "2.4  Sector Rotation (12-1 Month Momentum)",
        "Ranks sector ETFs by 12-month return skipping the most recent month (to avoid short-term reversal). Combines absolute momentum, relative strength vs SPY, and a mean-reversion penalty.",
        "Momentum_12_1 = Return(12 months) - Return(1 month)\nRisk_adj = Momentum / Volatility(63 days)\nRel_strength = Momentum - SPY_Momentum\nMR_penalty = -abs(1-month z-score) * 0.3\n\nScore = 0.4*risk_adj + 0.35*rel_strength + 0.25*MR_penalty\nLong top 3 sectors, short bottom 2",
        ["Documented factor (Moskowitz &amp; Grinblatt 1999)",
         "Skipping recent month avoids short-term reversal",
         "Low turnover: monthly rebalancing only"],
        ["Sector ETFs have higher correlation than individual stocks",
         "Factor crowding risk",
         "Underperforms in sector-correlated sell-offs"],
        "XLK (tech) 12M return +32%, skip month -2%, score = +30%. XLE (energy) -8%, score = -13%. Long XLK/XLY/XLF, short XLE/XLU."
    )

    story.append(PageBreak())

    # ── 3. MEAN REVERSION ──
    story.append(Paragraph("3. Mean-Reversion Models", s["SectionTitle"]))

    add_model(story, s,
        "3.1  Bollinger Band Mean Reversion",
        "Trades deviations from a rolling moving average, entering when price reaches the outer Bollinger Bands. Position size scales with degree of overshoot. Dynamic band width adjusts to volatility regime.",
        "MA = rolling_mean(price, window=21)\nStd = rolling_std(price, window=21)\nZ = (price - MA) / Std\n\nEntry: |Z| > entry_std (default 2.0)\nExit: |Z| < exit_std (default 0.5)\nDirection: short if Z > 0, long if Z < 0\nStrength = min((|Z| - entry) / entry, 1.0)",
        ["Simple and intuitive", "Dynamic bands adapt to current volatility",
         "Clear entry/exit rules reduce discretion"],
        ["Fails catastrophically in trends (price stays outside bands)",
         "Z-score assumes normal distribution (fat tails break this)",
         "No theoretical basis for band width choice"],
        "MSFT trades at $420, 21-day MA = $410, std = $8. Z = 1.25 (not yet at entry). Price spikes to $430: Z = 2.5, enter short. Reverts to $415: close for ~$15/share."
    )

    add_model(story, s,
        "3.2  Ornstein-Uhlenbeck (OU) Process",
        "A continuous-time mean-reverting stochastic process. Exon estimates OU parameters from discrete data via OLS regression on lagged values. The half-life determines trade horizon.",
        "dX = theta * (mu - X) * dt + sigma * dW\n\nDiscrete estimation:\n  dx_t = a + b * x_{t-1}  (OLS regression)\n  theta = -b  (mean reversion speed)\n  mu = -a / b  (long-run mean)\n  half_life = ln(2) / theta\n\nFilter: min_half_life <= half_life <= max_half_life",
        ["Theoretically grounded continuous-time stochastic process",
         "Half-life gives objective trade horizon",
         "Natural filter: rejects assets that don't mean-revert"],
        ["Assumes stationary process (breaks in trending markets)",
         "Parameter estimation sensitive to window length",
         "Small sample sizes produce unreliable estimates"],
        "BTC-ETH spread has half-life = 18 hours. Current deviation: 2.3 sigma above mean. OU predicts reversion within 36 hours. Short the spread, exit when z-score drops below 0.5."
    )

    add_model(story, s,
        "3.3  Pairs Trading (Engle-Granger Cointegration)",
        "Tests if two price series share a long-run equilibrium (cointegration). If cointegrated, the spread is stationary and mean-reverting. Exon scans all pairs and ranks by cointegration strength.",
        "Step 1: Engle-Granger test\n  Y = alpha + beta * X + epsilon\n  Test epsilon for stationarity (ADF test)\n\nStep 2: Spread = Y - beta * X - alpha\n\nStep 3: Z-score trading\n  Z = (Spread - mean) / std\n  Long if Z < -entry, Short if Z > +entry\n\nHalf-life via AR(1): half_life = -ln(2) / ln(phi)",
        ["Statistically rigorous cointegration test",
         "Market-neutral: dollar-neutral long-short pairs",
         "Mean-reversion empirically validated for cointegrated pairs"],
        ["Cointegration can break down (structural changes)",
         "Hedge ratio estimation error introduces drift",
         "Requires sufficient history for reliable testing"],
        "KO and PEP test cointegrated (p=0.02). Hedge ratio beta=0.85. Spread z-score hits -2.5: long 100 KO, short 85 PEP. Reverts over 2 weeks for $320 profit."
    )

    story.append(PageBreak())

    # ── 4. SIGNAL PROCESSING ──
    story.append(Paragraph("4. Statistical &amp; Signal Processing Models", s["SectionTitle"]))

    add_model(story, s,
        "4.1  Kalman Filter (State-Space Model)",
        "A recursive Bayesian estimator for tracking a time-varying hedge ratio in pairs trading. Unlike static OLS, the Kalman filter adapts the hedge ratio in real-time. The innovation z-score serves as the trading signal.",
        "State: theta_t = theta_{t-1} + w_t,  w ~ N(0, Q)\nObs:   y_t = [1, x_t] * theta_t + v_t,  v ~ N(0, R)\n\nPredict: theta_pred = theta_{t-1}, P_pred = P_{t-1} + Q\nUpdate:\n  innovation = y_t - H * theta_pred\n  S = H * P_pred * H' + R\n  K = P_pred * H' / S  (Kalman gain)\n  theta_t = theta_pred + K * innovation\n\nSignal = innovation / sqrt(S)",
        ["Adaptive: hedge ratio updates every bar",
         "Handles non-stationary relationships",
         "Provides uncertainty estimate (covariance P)",
         "Optimal estimator for linear-Gaussian systems"],
        ["Sensitive to noise parameters Q and R",
         "Assumes linear relationship",
         "Can overfit to noise if Q is too large"],
        "Tracking BTC-ETH hedge ratio: static OLS says beta=15.2, Kalman tracks it from 14.8 to 15.7 over 30 days. Innovation z-score = -2.8: long the spread, close at z=0 for $450 profit."
    )

    add_model(story, s,
        "4.2  Discrete Wavelet Transform (DWT)",
        "Decomposes price returns into different frequency components using Haar wavelets. Each level captures a different time-scale. Momentum is computed per scale and combined with frequency-dependent weights.",
        "Haar wavelet decomposition (5 levels):\n  D1: 2-4 hour cycles    (noise)\n  D2: 4-8 hour cycles    (intraday)\n  D3: 8-16 hour cycles   (session-level)\n  D4: 16-32 hour cycles  (multi-day)\n  D5: 32-64 hour cycles  (weekly)\n  A5: 64+ hours          (macro trend)\n\nWeights: higher for lower frequencies (macro > noise)",
        ["Separates signal from noise at each time-scale",
         "Captures multi-timeframe momentum simultaneously",
         "No look-ahead bias (causal decomposition)"],
        ["Haar wavelet is simplest (may miss smoother patterns)",
         "Boundary effects at start/end of data",
         "Weight selection across scales is subjective"],
        "BTC returns: D1 (noise) random, D4 (multi-day) strong uptrend +0.7, A5 (macro) +0.4. Combined with D4/A5 weighted 2x, D1 at 0.5x. Net momentum: +0.55, go long."
    )

    add_model(story, s,
        "4.3  Hurst Exponent (Rescaled Range Analysis)",
        "Measures long-range dependence structure of a time series. Determines whether a series is trending (H>0.5), random walk (H=0.5), or mean-reverting (H<0.5). Used as a regime filter.",
        "For each lag L:\n  1. Divide series into blocks of size L\n  2. R = max(cumsum) - min(cumsum) of deviations\n  3. S = std(block)\n  4. Average R/S across blocks\n\nHurst H = slope of log(R/S) vs log(L)\nH < 0.5: mean-reverting | H = 0.5: random | H > 0.5: trending",
        ["Non-parametric: no distribution assumptions",
         "Captures long-memory effects",
         "Natural strategy selector: trend vs mean-revert"],
        ["Sensitive to data length (needs 100+ observations)",
         "R/S method has upward bias for small samples",
         "Binary threshold choice is arbitrary"],
        "AAPL 63-day Hurst = 0.62 (trending): enable momentum, disable mean-reversion. Next quarter Hurst = 0.38 (mean-reverting): switch to Bollinger/OU strategies."
    )

    add_model(story, s,
        "4.4  Principal Component Analysis (PCA)",
        "Decomposes asset returns into orthogonal factors. PC1 is typically the market factor. Residuals represent idiosyncratic alpha. PCA-based stat arb trades residuals back to zero.",
        "Z = (X - mean) / std  (standardise)\nPCA: Z = U * S * V'  (SVD)\nFactor returns = Z * V[:, :k]\nResiduals = Z - Reconstructed\n\nVariance threshold: select k s.t. cum_var >= 90%\nTrading signal: z-score of cumulative residual",
        ["Data-driven: discovers structure without assumptions",
         "Removes systematic risk (market, sector factors)",
         "Residuals are uncorrelated (clean alpha signals)"],
        ["Components have no economic interpretation a priori",
         "Sensitive to outliers",
         "Number of components must be chosen"],
        "15-stock universe: PC1 explains 62% variance (market). NVDA residual = +3.2 sigma. Short NVDA residual, expecting reversion. 2 weeks later, residual returns to 0."
    )

    story.append(PageBreak())

    # ── 5. REGIME & FACTOR ──
    story.append(Paragraph("5. Regime Detection &amp; Factor Models", s["SectionTitle"]))

    add_model(story, s,
        "5.1  Gaussian Mixture Model (GMM)",
        "Clusters market conditions into discrete regimes (low-vol trending, normal, high-vol crisis) using unsupervised ML. Features include return level, volatility, and momentum. Drives dynamic strategy allocation. Uses scikit-learn's GaussianMixture.",
        "Features F = [return, volatility, momentum]\nGMM: P(x) = sum_k( pi_k * N(x | mu_k, Sigma_k) )\n  k = 3 regimes, full covariance, EM algorithm\n\nRegime allocation multipliers:\n  Regime 0 (trending): momentum 1.5x, MR 0.5x\n  Regime 1 (normal):   momentum 1.0x, MR 1.2x\n  Regime 2 (high vol): momentum 0.5x, breakout 1.5x",
        ["Discovers regimes without pre-specifying rules",
         "Soft clustering: provides probabilities",
         "Enables dynamic strategy allocation (core RenTech insight)"],
        ["Number of regimes must be pre-specified",
         "Initialisation-dependent (local optima)",
         "Regime labels can swap between refits"],
        "SPY: return=+0.1%, vol=12%, mom=+5%. GMM: 70% regime 0 (trending). Allocator boosts momentum 1.5x, reduces mean-reversion 0.5x."
    )

    add_model(story, s,
        "5.2  Multi-Factor Model (IC-Weighted)",
        "Ranks assets on 6 orthogonal factors and combines using Information Coefficient (IC) adaptive weighting. IC = Spearman rank correlation between factor and subsequent returns.",
        "Six factors (cross-sectional z-scored):\n  1. Momentum: blended [72, 168, 336] windows\n  2. Short-term reversal: -return(12h)\n  3. Volatility: -std (low-vol anomaly)\n  4. Liquidity: -avg(|return|)\n  5. Autocorrelation: lag-1 correlation\n  6. Skewness: -skew\n\nIC = Spearman(factor_rank, forward_returns)\nWeight_i = IC_i / sum(|IC_j|)",
        ["Diversified alpha: 6 independent return drivers",
         "Adaptive: IC weighting increases profitable factor exposure",
         "Self-correcting: declining IC reduces factor allocation"],
        ["Many parameters to tune",
         "IC estimation noisy on short windows",
         "Factor correlations change over time"],
        "NVDA: momentum z=+1.8, reversal z=-0.3, low_vol z=-1.2. Recent ICs: momentum=0.08, reversal=0.12. IC-weighted composite = +0.37. Ranks 2nd: long with strength 0.8."
    )

    add_model(story, s,
        "5.3  Post-Earnings Announcement Drift (PEAD)",
        "Detects earnings-like events via abnormal price gaps and trades continuation drift. Academically documented: stocks that gap up on earnings continue drifting up for 60 days.",
        "Event detection:\n  baseline_vol = std(returns excl. event window)\n  z_score = max(|return|) / baseline_vol\n  Trigger if z > gap_threshold (default 2.5)\n\nDrift signal:\n  decay = 0.95 ^ bars_since_event\n  direction = sign(event_return)\n  strength = min(z/5, 1.0) * decay",
        ["Strong academic evidence (60+ years of drift)",
         "High conviction: event-driven with clear catalyst",
         "Exponential decay naturally reduces position"],
        ["Proxy detection: not all gaps are earnings",
         "Drift may have decayed in recent years",
         "Limited opportunities: only when gaps occur"],
        "AAPL gaps up 4.5% on earnings. Z = 3.75. Initial strength 0.75. Day 20: strength = 0.75 * 0.95^20 = 0.27. Position slowly unwound."
    )

    add_model(story, s,
        "5.4  Market Microstructure (OBV &amp; A/D Line)",
        "Analyses volume patterns to detect institutional accumulation or distribution before price moves. Combines OBV trend, A/D line, and relative volume.",
        "OBV = cumsum( sign(return) * volume )\nOBV_trend = slope(OBV, window) / std(OBV)\n\nA/D = cumsum( CLV * volume )\nCLV = (2*close - high - low) / (high - low)\n\nRVOL = volume / rolling_mean(volume, 63d)\nSignal = OBV_trend * RVOL_boost",
        ["Leading indicator: volume precedes price",
         "Detects institutional activity",
         "RVOL confirmation reduces false signals"],
        ["Volume data not always available",
         "Falls back to price-only proxy (less accurate)",
         "Divergences can persist longer than expected"],
        "GOOGL: price flat, OBV trending up for 2 weeks (z=2.1). RVOL=1.8. Accumulation signal: long GOOGL. Price breaks out to $185."
    )

    story.append(PageBreak())

    # ── 6. PORTFOLIO OPTIMISATION ──
    story.append(Paragraph("6. Portfolio Optimisation Models", s["SectionTitle"]))

    add_model(story, s,
        "6.1  Mean-Variance Optimisation (Markowitz)",
        "Harry Markowitz's 1952 framework: maximise expected return for given risk. Uses quadratic programming with risk aversion parameter gamma.",
        "Maximise: w'*mu - (gamma/2) * w'*Sigma*w\nSubject to: sum(w_i) = 1, |w_i| <= max_weight\nSolver: scipy SLSQP\ngamma = risk aversion (default 1.0)",
        ["Theoretically optimal for quadratic utility",
         "Considers correlations (diversification benefit)",
         "Industry standard for institutional allocation"],
        ["Sensitive to input estimation errors ('error maximiser')",
         "Concentrated portfolios from small covariance perturbations",
         "Assumes normally distributed returns"],
        "5-asset portfolio: gamma=1.0 gives [15%, 20%, 10%, 20%, 15%] with Sharpe 1.8. Gamma=3.0 gives more diversified [18%, 15%, 20%, 18%, 19%] with Sharpe 1.5."
    )

    add_model(story, s,
        "6.2  Risk Parity",
        "Allocates capital so each asset contributes equally to total portfolio risk. Ignores expected returns and focuses purely on risk diversification.",
        "Portfolio var: sigma_p^2 = w' * Sigma * w\nMarginal risk: MR_i = (Sigma @ w)_i\nRisk contribution: RC_i = w_i * MR_i / sigma_p\n\nObjective: minimise sum( (RC_i/sum(RC) - 1/N)^2 )\nConstraint: w_i >= 0",
        ["No return estimates needed (avoids estimation error)",
         "True diversification by risk, not capital",
         "Robust and less sensitive to input perturbations"],
        ["Ignores expected returns (may allocate to poor assets)",
         "Tends to overweight low-vol assets",
         "Implicitly assumes equal Sharpe ratios"],
        "AAPL (vol=25%), TLT (vol=12%), GLD (vol=15%). Equal capital = 33% each, but AAPL contributes 52% risk. Risk parity: AAPL 19%, TLT 42%, GLD 39%."
    )

    add_model(story, s,
        "6.3  Kelly Criterion",
        "Optimal bet sizing that maximises long-run geometric growth rate. Exon uses fractional Kelly (default 25%) for safety.",
        "Full Kelly: w = Sigma^{-1} @ mu\nFractional Kelly: w_frac = 0.25 * w_full\nMax weight: clipped to +/- 0.25\nNormalised to sum(|w|) = 1",
        ["Maximises long-run wealth growth (theoretically optimal)",
         "Naturally sizes bigger on high-Sharpe opportunities",
         "Fractional Kelly reduces drawdown risk"],
        ["Full Kelly is extremely aggressive",
         "Requires accurate return/covariance estimates",
         "Sensitive to estimation error"],
        "Strategy with E[r]=15%, vol=20%: Full Kelly = 3.75x leverage. Quarter Kelly = 0.94x. Kelly allocates more to NVDA (Sharpe 1.2, 30%) vs INTC (Sharpe 0.3, 8%)."
    )

    story.append(PageBreak())

    # ── 7. RISK MANAGEMENT ──
    story.append(Paragraph("7. Risk Management Models", s["SectionTitle"]))

    add_model(story, s,
        "7.1  Value at Risk (VaR) &amp; Expected Shortfall (CVaR)",
        "VaR measures worst expected loss at a given confidence level. CVaR measures average loss in the worst cases beyond VaR. Both are portfolio risk constraints in Exon.",
        "VaR_95 = percentile(portfolio_returns, 5th)\nVaR_99 = percentile(portfolio_returns, 1st)\nCVaR = mean(returns where returns <= VaR_95)\n\nConstraint: |VaR_95| <= 2% of portfolio\nIf breached: reduce exposure proportionally",
        ["Industry standard risk metric (Basel III)",
         "CVaR captures tail risk beyond VaR",
         "Historical method: no distribution assumptions"],
        ["Historical VaR assumes past distribution continues",
         "VaR is not sub-additive",
         "Short history underestimates tail risk"],
        "10-stock portfolio, 252 days. VaR_95 = -1.8%, CVaR = -3.2%. If VaR exceeds 2% limit, reduce all positions by VaR/limit ratio."
    )

    add_model(story, s,
        "7.2  Volatility Targeting",
        "Scales position size to target specific annualised volatility. If realised vol rises, position is reduced to maintain constant risk.",
        "realised_vol = std(returns, lookback) * sqrt(bars_per_year)\nscale = target_vol / realised_vol\nadjusted_position = raw_position * min(scale, max_scale)\n\nDefault: target_vol = 10-15% annualised",
        ["Equalises risk across assets with different volatilities",
         "Automatically de-levers in volatile markets",
         "Simple and effective: used by most quant funds"],
        ["Backward-looking: uses past vol for future sizing",
         "Lag in responding to sudden vol spikes",
         "Can force exit at worst time (selling during crash)"],
        "BTC vol=60%, target=15%: scale=0.25, hold 25%. AAPL vol=20%, target=15%: scale=0.75, hold 75%. Both contribute ~15% vol."
    )

    add_model(story, s,
        "7.3  Drawdown Circuit Breaker",
        "Automatically reduces portfolio exposure when drawdown from peak exceeds threshold. Prevents catastrophic losses by progressive de-leveraging.",
        "peak_equity = max(equity_curve)\ndrawdown = (current - peak) / peak\n\nIf drawdown < -max_dd_pct:\n  reduction = max_dd / drawdown\n  weights *= reduction\n\nDefault: max_dd_pct = 15%, min_cash = 5%",
        ["Prevents ruin: caps maximum loss from peak",
         "Automatic: no discretionary intervention needed",
         "Progressive: proportional to drawdown depth"],
        ["Pro-cyclical: sells into falling markets",
         "Can prevent recovery (reduced positions miss rebound)",
         "Frequent triggering creates whipsaw"],
        "Portfolio peaks at $120K, falls to $100K (dd=-16.7%). Breaches 15%. Reduction = 0.90. All positions scaled 0.90x."
    )

    story.append(PageBreak())

    # ── 8. BACKTESTING ──
    story.append(Paragraph("8. Backtesting &amp; Validation", s["SectionTitle"]))

    add_model(story, s,
        "8.1  Walk-Forward Analysis",
        "Rolling out-of-sample testing that avoids overfitting. Strategy is trained on each window and tested on subsequent unseen data. Results aggregated across all out-of-sample windows.",
        "For each window i:\n  Train: data[start : start + train_size]\n  Test:  data[start + train : start + train + test_size]\n  Step:  start += step\n\nEquities: train=252d, test=63d, step=63d\nCrypto:   train=720h, test=168h, step=168h\n\nMetrics: mean Sharpe, % positive windows, worst DD",
        ["True out-of-sample testing (no look-ahead bias)",
         "Detects overfitting: must work on unseen data",
         "Multiple windows give statistical significance"],
        ["Reduces total test data",
         "Short test windows have noisy estimates",
         "Doesn't capture all regimes in each window"],
        "2 years daily data: 4 out-of-sample windows. Mean Sharpe 1.2 (consistent). One window -0.5 signals regime sensitivity. 75% positive Sharpe."
    )

    add_model(story, s,
        "8.2  Monte Carlo Sharpe Significance Test",
        "Bootstrap hypothesis test: is the Sharpe ratio statistically significant or from random chance? Shuffles returns to destroy signal and computes null distribution.",
        "1. Compute actual_sharpe from real returns\n2. For i in 1..10000:\n   - Shuffle returns (destroy temporal structure)\n   - Compute null_sharpe_i\n3. p_value = fraction(null_sharpes >= actual_sharpe)\n\nSignificant if p < 0.05",
        ["Non-parametric: no distribution assumptions",
         "Directly answers: 'is this Sharpe real or luck?'",
         "Accounts for return characteristics (kurtosis, skew)"],
        ["Computationally expensive (10K permutations)",
         "Destroys autocorrelation (may overstate significance for momentum)",
         "Doesn't account for multiple testing"],
        "Strategy Sharpe=2.1. After 10K shuffles, null mean=0.01, std=0.35. p=0.001 (highly significant, not luck)."
    )

    add_model(story, s,
        "8.3  Options Backtesting Engine",
        "Specialised backtester for options: Black-Scholes repricing, daily Greeks tracking, IV path simulation with mean-reverting dynamics, expiry settlement, and position lifecycle management.",
        "IV simulation (per asset, per day):\n  IV_t = IV_{t-1} + k*(RV*1.1 - IV_{t-1}) + s_v*noise\n  k=0.03 (mean reversion), s_v=0.01 (vol of vol)\n\nDaily mark-to-market:\n  price = BS(spot_t, strike, DTE_t, IV_t)\n  Greeks = compute_greeks(...)\n\nExpiry: max(spot-strike, 0) for calls * multiplier",
        ["Realistic: theta decay, IV changes, gamma all modelled",
         "Greeks tracking enables risk-aware development",
         "Handles multi-leg structures (spreads, condors)"],
        ["Simulated IV paths are approximations",
         "No volatility smile modelling (flat IV)",
         "No early exercise (European-style only)"],
        "Momentum on AAPL -> bull call spread (high IV). Buy 150C, sell 155C, debit $2.50. Over 30 days: spot +$8, IV -3%. BS reprice: spread $4.20. Profit $1.70/contract."
    )

    story.append(PageBreak())

    # ── 9. ML MODELS ──
    story.append(Paragraph("9. Machine Learning Models (Available)", s["SectionTitle"]))
    story.append(Paragraph(
        "XGBoost and LightGBM are included as optional dependencies for future strategy development. "
        "They are gradient boosting frameworks commonly used in quantitative finance for return prediction, "
        "feature importance, and non-linear signal combination.",
        s["Body"]
    ))

    add_model(story, s,
        "9.1  XGBoost (eXtreme Gradient Boosting)",
        "A gradient boosting ensemble by Tianqi Chen (2016). Builds additive decision trees where each tree corrects errors of the ensemble. Widely used in quant finance for return prediction, alpha combination, and regime classification.",
        "Objective: min sum(L(y_i, y_hat_i)) + sum(Omega(f_k))\n  L = loss (MSE or logloss)\n  Omega = regularisation (tree complexity)\n\ny_hat^(t) = y_hat^(t-1) + eta * f_t(x_i)\nf_t = tree fitted to negative gradient of loss\n\nKey params: max_depth(3-8), n_estimators(100-1000),\nlearning_rate(0.01-0.1), subsample, colsample_bytree",
        ["Handles non-linear feature interactions automatically",
         "Built-in regularisation prevents overfitting",
         "Feature importance identifies useful signals",
         "Handles missing data natively",
         "Fast training with GPU support"],
        ["Black box: hard to interpret predictions",
         "Prone to overfitting on financial data (low SNR)",
         "Requires careful time-series cross-validation",
         "Feature engineering still required",
         "Non-stationary markets violate i.i.d. assumption"],
        "Features: [momentum_5d, vol_ratio, OBV_z, RSI, IV_rank]. Target: 5-day return direction. XGBClassifier(max_depth=4, n_estimators=200). Walk-forward accuracy: 54% (profitable after costs if consistent)."
    )

    add_model(story, s,
        "9.2  LightGBM (Light Gradient Boosting Machine)",
        "Microsoft's gradient boosting framework (2017), optimised for speed and memory. Uses leaf-wise tree growth, histogram binning, and GOSS (Gradient-based One-Side Sampling) for faster training on large datasets.",
        "Key differences from XGBoost:\n  1. Leaf-wise growth (vs level-wise): faster convergence\n  2. Histogram binning: reduces memory and split time\n  3. GOSS: keeps large-gradient instances, downsamples small\n  4. EFB: bundles mutually exclusive features\n\nKey params: num_leaves(31), learning_rate,\nn_estimators, min_data_in_leaf, feature_fraction",
        ["10-20x faster training than XGBoost on large data",
         "Lower memory usage (histogram-based)",
         "Native categorical feature support",
         "Often higher accuracy with less tuning"],
        ["Leaf-wise growth overfits more on small datasets",
         "Same fundamental limitations as all boosting",
         "Histogram binning loses precision for continuous features",
         "Requires careful num_leaves tuning"],
        "Same features, 10 years hourly data (87,600 samples). LightGBM trains in 12s vs XGBoost 180s. LGBMRegressor predicts next-day returns. Walk-forward IC = 0.04 (small but consistent alpha)."
    )

    story.append(PageBreak())

    # ── 10. IV SURFACE ──
    story.append(Paragraph("10. IV Surface &amp; Volatility Models", s["SectionTitle"]))

    add_model(story, s,
        "10.1  IV Rank &amp; IV Percentile",
        "Two measures of where current IV sits relative to history. IV Rank uses min/max, IV Percentile counts observations below current. Both drive the signal-to-options structure mapping.",
        "IV Rank = (current_IV - 52wk_low) / (52wk_high - 52wk_low)\nIV Percentile = count(historical_IV < current) / N\n\nClassification:\n  IV < 30th pctile: LOW (buy premium)\n  IV 30-70th pctile: NORMAL\n  IV > 70th pctile: HIGH (sell premium)",
        ["Simple and interpretable",
         "Historical context for current IV level",
         "Direct mapping to trading strategy"],
        ["Backward-looking: past range may not predict future",
         "IV Rank distorted by single extreme spike",
         "Doesn't capture term structure or skew"],
        "AAPL ATM IV=28%, 52wk range 18%-45%. IV Rank=37% (low-normal). IV Percentile=42%. For bullish signal: select bull call spread."
    )

    add_model(story, s,
        "10.2  Variance Risk Premium (VRP)",
        "The difference between implied and realised volatility. Persistently positive VRP means options are systematically overpriced, creating a structural edge for premium sellers.",
        "VRP = IV - RV\n\nRealised Vol: RV = std(returns, window) * sqrt(252)\nMulti-window: 5d, 21d, 63d\nEWMA: exponentially weighted (recent emphasis)\n\nIV Proxy: max(rv_5d, rv_21d, rv_63d, ewma) * 1.1\n\nVRP > 0: options expensive (sell premium)\nVRP < 0: options cheap (buy premium, rare)",
        ["Well-documented structural premium",
         "Positive VRP = long-run edge for sellers",
         "Multiple RV windows capture different regimes"],
        ["VRP can go negative during crises",
         "Selling vol has tail risk (earned slowly, lost fast)",
         "IV proxy is approximate without live data"],
        "SPY IV=22%, 21d RV=15%. VRP=+7% (options expensive). Iron condor has structural edge. But March 2020: VRP=-30%. Use defined-risk structures."
    )

    add_model(story, s,
        "10.3  Signal-to-Options Structure Mapping",
        "The core decision engine mapping equity signals (direction + strength) and IV regime into 12 options structures. Bridges equity strategy layer to options execution.",
        "Strong bullish + Low IV  -> Long call\nStrong bullish + High IV -> Bull call spread\nStrong bearish + Low IV  -> Long put\nStrong bearish + High IV -> Bear put spread\nWeak signal + High IV    -> Iron condor\nWeak signal + Low IV     -> Long strangle\nBreakout + Low IV        -> Long straddle\nMean-reversion + High IV -> Short strangle\nMedium bull + High IV    -> Bull put spread\nMedium bear + High IV    -> Bear call spread\n\nConviction: strong > 0.6, weak < 0.25",
        ["Systematic: removes emotional bias",
         "IV-regime-aware: buys cheap, sells expensive",
         "12 structures cover all views"],
        ["Simplified: real traders consider more factors",
         "Fixed thresholds may not suit all conditions",
         "No individual Greek limits per structure"],
        "MSFT signal: direction=+0.8, strength=0.7 (strong bullish). IV pctile=75% (high). Mapper: bull call spread. Buy 420C, sell 440C, 45 DTE. Max loss $5.50, max gain $14.50. Risk-reward 2.6:1."
    )

    story.append(PageBreak())

    # ── APPENDIX: SUMMARY TABLE ──
    story.append(Paragraph("Appendix: Model Summary Table", s["SectionTitle"]))

    header = ["Model", "Category", "Primary Use", "Source File"]
    rows = [
        ["Black-Scholes", "Options Pricing", "Mark-to-market, premium", "options/pricing.py"],
        ["Greeks (d/g/t/v/r)", "Options Risk", "Sensitivity analysis", "options/pricing.py"],
        ["Implied Vol (Brent)", "Options Pricing", "IV extraction", "options/pricing.py"],
        ["Put-Call Parity", "Options Validation", "Data quality check", "options/pricing.py"],
        ["Time-Series Momentum", "Trend Following", "Directional trading", "strategies/momentum.py"],
        ["Cross-Sectional Mom.", "Relative Value", "Long-short ranking", "strategies/momentum.py"],
        ["Adaptive Momentum", "Trend Following", "Regime-adaptive", "strategies/momentum.py"],
        ["Sector Rotation", "Factor Rotation", "Sector ETF allocation", "strategies/sector_rot.py"],
        ["Bollinger Mean Rev.", "Mean Reversion", "Range-bound trading", "strategies/mean_rev.py"],
        ["Ornstein-Uhlenbeck", "Mean Reversion", "Half-life estimation", "strategies/mean_rev.py"],
        ["Engle-Granger", "Pairs Trading", "Spread trading", "research/coint.py"],
        ["Kalman Filter", "State-Space", "Dynamic hedge ratios", "strategies/kalman.py"],
        ["Wavelet DWT", "Signal Processing", "Multi-scale momentum", "strategies/wavelet.py"],
        ["Hurst Exponent", "Regime Detection", "Trend/MR filter", "strategies/hurst.py"],
        ["PCA", "Factor Model", "Systematic risk removal", "research/pca.py"],
        ["GMM", "Regime Detection", "Market regime cluster", "research/regime.py"],
        ["Multi-Factor (IC)", "Factor Model", "Cross-sect. alpha", "strategies/multifactor.py"],
        ["PEAD", "Event-Driven", "Earnings drift", "strategies/earnings.py"],
        ["OBV / A-D Line", "Microstructure", "Volume flow", "strategies/micro.py"],
        ["Mean-Variance", "Portfolio Opt.", "Optimal allocation", "portfolio/optimizer.py"],
        ["Risk Parity", "Portfolio Opt.", "Equal risk alloc.", "portfolio/optimizer.py"],
        ["Kelly Criterion", "Position Sizing", "Growth-optimal", "portfolio/optimizer.py"],
        ["VaR / CVaR", "Risk Mgmt", "Tail risk monitoring", "risk/manager.py"],
        ["Vol Targeting", "Risk Mgmt", "Risk normalisation", "risk/manager.py"],
        ["Walk-Forward", "Validation", "OOS testing", "backtest/engine.py"],
        ["Monte Carlo Sharpe", "Validation", "Significance test", "backtest/metrics.py"],
        ["IV Rank/Percentile", "Vol Analysis", "IV regime class.", "options/vol_surface.py"],
        ["Variance Risk Prem.", "Vol Analysis", "Premium edge", "options/vol_surface.py"],
        ["Signal-Structure Map", "Options Strategy", "Structure select", "options/strat_mapper.py"],
        ["XGBoost", "Machine Learning", "Return prediction", "pyproject.toml (dep)"],
        ["LightGBM", "Machine Learning", "Fast boosting", "pyproject.toml (dep)"],
    ]

    col_widths = [95, 80, 95, 95]
    table_data = [header] + rows

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#14397A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#F5F5F5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ])
    tbl.setStyle(tbl_style)
    story.append(tbl)

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<i>This document covers 31 quantitative models implemented across the Exon Research platform. "
        "All models are backtested with walk-forward validation and Monte Carlo significance testing. "
        "The platform runs 260 automated tests across equity, crypto, and options modules. "
        "XGBoost and LightGBM are available as optional dependencies for ML-based strategy development.</i>",
        s["Body"]
    ))

    doc.build(story)
    print("PDF generated: Exon_Research_Models_Reference.pdf")


if __name__ == "__main__":
    build_pdf()
