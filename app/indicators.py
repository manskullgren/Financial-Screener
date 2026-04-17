import numpy as np
import pandas as pd
from scipy import stats


async def calculate_rsi(price_history: pd.Series, period: int = 14) -> float:
    if len(price_history) < period + 1:
        return 50.0

    deltas = price_history.diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)

    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()

    for i in range(period + 1, len(price_history)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gains.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + losses.iloc[i]) / period

    # Calculate final RSI
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else 100
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk using multiple methods."""

    historical_var = np.percentile(returns, (1 - confidence) * 100)

    mean_return = returns.mean()
    std_return = returns.std()
    z_score = stats.norm.ppf(1 - confidence)
    parametric_var = mean_return + z_score * std_return

    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    cf_adjustment = z_score + (z_score**2 - 1) * skewness / 6
    cf_var = mean_return + cf_adjustment * std_return

    return min(historical_var, parametric_var, cf_var)
