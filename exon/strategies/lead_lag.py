"""Lead-lag exploitation strategy for crypto markets.

The single most Renaissance-like edge in crypto: large-cap assets
(BTC, ETH) consistently lead smaller-cap altcoins by 1-6 hours.
This temporal structure arises because:

1. Institutional flow hits BTC/ETH first (deepest liquidity)
2. Retail and algorithmic arbitrage propagates to alts with delay
3. Cross-exchange latency amplifies the lag on Coinbase vs derivatives

This strategy estimates the lead-lag structure via cross-correlation,
then trades the lagging assets in the direction of the leader's recent
move, capturing the delayed propagation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Signal, Strategy


class LeadLagExploitation(Strategy):
    """Trade altcoins based on lagged response to BTC/ETH moves.

    Dynamically estimates the optimal lag for each asset pair via
    rolling cross-correlation, then generates signals for lagging
    assets when the leader has moved but the follower hasn't yet.
    """

    name = "lead_lag"

    def __init__(
        self,
        leaders: list[str] | None = None,
        max_lag: int = 6,
        xcorr_window: int = 168,
        min_correlation: float = 0.15,
        signal_window: int = 6,
        decay_hours: int = 12,
        vol_target: float = 0.15,
    ):
        """
        Parameters
        ----------
        leaders : assets to use as leaders; defaults to BTC-USD, ETH-USD
        max_lag : maximum lag (in bars) to test in cross-correlation
        xcorr_window : rolling window for estimating lead-lag structure
        min_correlation : minimum lagged correlation to consider tradeable
        signal_window : lookback (in bars) for the leader's recent move
        decay_hours : how quickly the signal decays after the leader moves
        vol_target : annualised volatility target for position sizing
        """
        self.leaders = leaders or ["BTC-USD", "ETH-USD"]
        self.max_lag = max_lag
        self.xcorr_window = xcorr_window
        self.min_correlation = min_correlation
        self.signal_window = signal_window
        self.decay_hours = decay_hours
        self.vol_target = vol_target

    def generate_signals(self, data: pd.DataFrame, **kwargs) -> list[Signal]:
        returns = np.log(data / data.shift(1)).dropna()
        timestamp = data.index[-1]

        if len(returns) < self.xcorr_window + self.max_lag:
            return []

        # Identify which leaders are present in the data
        available_leaders = [l for l in self.leaders if l in returns.columns]
        if not available_leaders:
            return []

        followers = [c for c in returns.columns if c not in available_leaders]
        if not followers:
            return []

        signals = []

        for leader in available_leaders:
            leader_ret = returns[leader]
            # Leader's recent move (z-scored)
            recent_leader_ret = leader_ret.iloc[-self.signal_window :].sum()
            leader_vol = leader_ret.iloc[-self.xcorr_window :].std()
            leader_z = (
                recent_leader_ret / (leader_vol * np.sqrt(self.signal_window))
                if leader_vol > 0
                else 0
            )

            if abs(leader_z) < 0.5:
                continue  # leader hasn't moved enough to propagate

            for follower in followers:
                follower_ret = returns[follower]

                # Estimate optimal lag via cross-correlation
                best_lag, best_corr = self._find_optimal_lag(
                    leader_ret.iloc[-self.xcorr_window :],
                    follower_ret.iloc[-self.xcorr_window :],
                )

                if best_corr < self.min_correlation or best_lag == 0:
                    continue  # no exploitable lag relationship

                # Check if the follower has already caught up
                follower_recent = follower_ret.iloc[-best_lag :].sum() if best_lag > 0 else 0
                follower_vol = follower_ret.iloc[-self.xcorr_window :].std()
                follower_z = (
                    follower_recent / (follower_vol * np.sqrt(max(best_lag, 1)))
                    if follower_vol > 0
                    else 0
                )

                # Signal = leader moved, follower hasn't caught up yet
                gap = leader_z * best_corr - follower_z
                if abs(gap) < 0.3:
                    continue  # follower already caught up

                direction = np.sign(gap)

                # Strength based on: correlation quality, gap size, decay
                corr_score = min(best_corr / 0.5, 1.0)
                gap_score = min(abs(gap) / 2.0, 1.0)

                # Vol targeting
                annual_vol = follower_vol * np.sqrt(8760)
                vol_scale = min(self.vol_target / annual_vol, 2.0) if annual_vol > 0 else 0

                strength = min(corr_score * gap_score * vol_scale, 1.0)

                if strength > 0.05:
                    signals.append(
                        Signal(
                            timestamp=timestamp,
                            asset=follower,
                            direction=direction,
                            strength=strength,
                            metadata={
                                "leader": leader,
                                "optimal_lag": best_lag,
                                "lagged_corr": best_corr,
                                "leader_z": leader_z,
                                "follower_z": follower_z,
                                "gap": gap,
                            },
                        )
                    )

        # Deduplicate: if multiple leaders signal the same follower, average
        return self._deduplicate(signals, timestamp)

    def _find_optimal_lag(
        self,
        leader: pd.Series,
        follower: pd.Series,
    ) -> tuple[int, float]:
        """Find the lag that maximises cross-correlation.

        Tests lags from 1 to max_lag (leader leads follower).
        Returns (best_lag, best_correlation).
        """
        best_lag = 0
        best_corr = 0.0

        for lag in range(1, self.max_lag + 1):
            shifted_leader = leader.shift(lag).dropna()
            aligned_follower = follower.reindex(shifted_leader.index)
            common = shifted_leader.dropna().index.intersection(aligned_follower.dropna().index)
            if len(common) < 50:
                continue
            corr = shifted_leader.loc[common].corr(aligned_follower.loc[common])
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        return best_lag, abs(best_corr)

    @staticmethod
    def _deduplicate(signals: list[Signal], timestamp) -> list[Signal]:
        """If multiple leaders signal the same follower, blend them."""
        by_asset: dict[str, list[Signal]] = {}
        for s in signals:
            by_asset.setdefault(s.asset, []).append(s)

        deduped = []
        for asset, sigs in by_asset.items():
            if len(sigs) == 1:
                deduped.append(sigs[0])
            else:
                # Weighted average by strength
                total_w = sum(s.strength for s in sigs)
                if total_w == 0:
                    continue
                blended_dir = sum(s.direction * s.strength for s in sigs) / total_w
                avg_strength = total_w / len(sigs)
                leaders = [s.metadata.get("leader", "?") for s in sigs]
                deduped.append(
                    Signal(
                        timestamp=timestamp,
                        asset=asset,
                        direction=np.sign(blended_dir),
                        strength=min(avg_strength, 1.0),
                        metadata={"leaders": leaders, "n_confirmations": len(sigs)},
                    )
                )
        return deduped
