import pandas as pd


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
