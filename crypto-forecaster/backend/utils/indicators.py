"""Technical indicators calculator for cryptocurrency data."""
import pandas as pd
import numpy as np
from typing import Dict


class TechnicalIndicators:
    """Calculate technical indicators for cryptocurrency price data."""

    @staticmethod
    def calculate_sma(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=window, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band
        }

    @classmethod
    def calculate_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a dataframe."""
        df = df.copy()

        # Ensure data is sorted by date
        df = df.sort_values("date")

        # Moving Averages
        df["sma_7"] = cls.calculate_sma(df["close"], 7)
        df["sma_30"] = cls.calculate_sma(df["close"], 30)
        df["ema_12"] = cls.calculate_ema(df["close"], 12)
        df["ema_26"] = cls.calculate_ema(df["close"], 26)

        # RSI
        df["rsi"] = cls.calculate_rsi(df["close"])

        # MACD
        macd_data = cls.calculate_macd(df["close"])
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_histogram"] = macd_data["histogram"]

        # Bollinger Bands
        bollinger = cls.calculate_bollinger_bands(df["close"])
        df["bollinger_high"] = bollinger["upper"]
        df["bollinger_mid"] = bollinger["middle"]
        df["bollinger_low"] = bollinger["lower"]

        # Volume SMA
        df["volume_sma"] = cls.calculate_sma(df["volume"], 30)

        return df
