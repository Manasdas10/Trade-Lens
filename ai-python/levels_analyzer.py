"""
levels_analyzer.py — Spot Trade Levels Analytics Engine
Computes support/resistance pivot levels, precise buy/sell entry zones,
stop losses, targets, and handles multi-timeframe trend and level synthesis.
"""

import numpy as np
import pandas as pd
import math

# =====================================================================
# 1. PIVOT POINTS CALCULATOR (SUPPORT / RESISTANCE)
# =====================================================================

def calculate_pivot_points(df):
    """
    Computes Classic, Fibonacci, and Camarilla Pivot Points.
    Uses the latest completed bar to calculate levels for the current/upcoming bar.
    """
    if df.empty or len(df) < 2:
        return {}

    # Use the last fully closed candle for calculations
    last_row = df.iloc[-1]
    high = float(last_row["high"])
    low = float(last_row["low"])
    close = float(last_row["close"])
    
    # Classic Pivot Points
    pivot_classic = (high + low + close) / 3.0
    r1_classic = (2.0 * pivot_classic) - low
    s1_classic = (2.0 * pivot_classic) - high
    r2_classic = pivot_classic + (high - low)
    s2_classic = pivot_classic - (high - low)
    r3_classic = high + 2.0 * (pivot_classic - low)
    s3_classic = low - 2.0 * (high - pivot_classic)

    # Fibonacci Pivot Points
    range_val = high - low
    pivot_fib = (high + low + close) / 3.0
    r1_fib = pivot_fib + (0.382 * range_val)
    s1_fib = pivot_fib - (0.382 * range_val)
    r2_fib = pivot_fib + (0.618 * range_val)
    s2_fib = pivot_fib - (0.618 * range_val)
    r3_fib = pivot_fib + (1.000 * range_val)
    s3_fib = pivot_fib - (1.000 * range_val)

    # Camarilla Pivot Points
    r1_cam = close + (range_val * 1.1 / 12.0)
    s1_cam = close - (range_val * 1.1 / 12.0)
    r2_cam = close + (range_val * 1.1 / 6.0)
    s2_cam = close - (range_val * 1.1 / 6.0)
    r3_cam = close + (range_val * 1.1 / 4.0)
    s3_cam = close - (range_val * 1.1 / 4.0)
    r4_cam = close + (range_val * 1.1 / 2.0)  # Breakout level
    s4_cam = close - (range_val * 1.1 / 2.0)  # Breakdown level

    return {
        "classic": {
            "R3": r3_classic, "R2": r2_classic, "R1": r1_classic,
            "P": pivot_classic,
            "S1": s1_classic, "S2": s2_classic, "S3": s3_classic
        },
        "fibonacci": {
            "R3": r3_fib, "R2": r2_fib, "R1": r1_fib,
            "P": pivot_fib,
            "S1": s1_fib, "S2": s2_fib, "S3": s3_fib
        },
        "camarilla": {
            "R4": r4_cam, "R3": r3_cam, "R2": r2_cam, "R1": r1_cam,
            "S1": s1_cam, "S2": s2_cam, "S3": s3_cam, "S4": s4_cam
        }
    }


# =====================================================================
# 2. SPOT TRADE RECOMMENDATION GENERATOR
# =====================================================================

def get_trade_setup(spot_price, trend_signal, atr, levels):
    """
    Generates actionable buy/sell execution levels (entry zones, stop loss, targets).
    
    Parameters
    ----------
    spot_price : float
        Current spot price.
    trend_signal : str
        "BUY", "SELL", or "HOLD".
    atr : float
        Average True Range (volatility measure).
    levels : dict
        Calculated pivot levels.
        
    Returns
    -------
    dict
        Execution levels details.
    """
    # Use Camarilla pivots for tight intraday levels, fallback to Classic
    pivot_type = "camarilla" if "camarilla" in levels else "classic"
    p_lvls = levels[pivot_type]
    
    # Defaults
    buy_entry_min = spot_price - 0.5 * atr
    buy_entry_max = spot_price + 0.2 * atr
    sell_entry_min = spot_price - 0.2 * atr
    sell_entry_max = spot_price + 0.5 * atr
    sl = spot_price - 1.5 * atr
    t1 = spot_price + 1.5 * atr
    t2 = spot_price + 3.0 * atr
    rr_ratio = 1.5
    rationale = "No strong trend detected. Monitor pivot boundaries."

    if trend_signal == "BUY":
        # Buy Zone: Buy pullback to support (e.g. S1/S2) or on breakout above R1/P
        s1 = p_lvls.get("S1", spot_price - 0.5 * atr)
        p = p_lvls.get("P", spot_price)
        
        # Bullish Entry: Buy zone between S1 and slightly above P
        buy_entry_min = min(s1, spot_price - 0.3 * atr)
        buy_entry_max = max(p, spot_price + 0.1 * atr)
        
        # Stop Loss: Set below S2 or 1.5 ATR
        s2 = p_lvls.get("S2", spot_price - 1.5 * atr)
        sl = min(s2, spot_price - 1.2 * atr)
        
        # Targets: T1 at R1/R2, T2 at R3
        t1 = p_lvls.get("R1", spot_price + 1.2 * atr)
        t2 = p_lvls.get("R3", spot_price + 2.5 * atr)
        
        # Ensure correct ordering
        if sl >= buy_entry_min:
            sl = buy_entry_min - 1.0 * atr
        if t1 <= buy_entry_max:
            t1 = buy_entry_max + 1.0 * atr
        if t2 <= t1:
            t2 = t1 + 1.5 * atr
            
        risk = buy_entry_max - sl
        reward = t1 - buy_entry_max
        rr_ratio = reward / max(1e-5, risk)
        
        rationale = (
            f"Bullish trend confirmed. Optimal BUY entry zone is between "
            f"{buy_entry_min:,.2f} (pullback) and {buy_entry_max:,.2f} (momentum breakout). "
            f"Set Stop Loss below major support level at {sl:,.2f}."
        )
        
    elif trend_signal == "SELL":
        # Sell/Short Zone: Sell pullback to resistance (e.g. R1/R2) or breakdown below S1/P
        r1 = p_lvls.get("R1", spot_price + 0.5 * atr)
        p = p_lvls.get("P", spot_price)
        
        # Bearish Entry: Short zone between P and R1
        sell_entry_min = min(p, spot_price - 0.1 * atr)
        sell_entry_max = max(r1, spot_price + 0.3 * atr)
        
        # Stop Loss: Set above R2 or 1.5 ATR
        r2 = p_lvls.get("R2", spot_price + 1.5 * atr)
        sl = max(r2, spot_price + 1.2 * atr)
        
        # Targets: T1 at S1/S2, T2 at S3
        t1 = p_lvls.get("S1", spot_price - 1.2 * atr)
        t2 = p_lvls.get("S3", spot_price - 2.5 * atr)
        
        # Ensure correct ordering
        if sl <= sell_entry_max:
            sl = sell_entry_max + 1.0 * atr
        if t1 >= sell_entry_min:
            t1 = sell_entry_min - 1.0 * atr
        if t2 >= t1:
            t2 = t1 - 1.5 * atr
            
        risk = sl - sell_entry_min
        reward = sell_entry_min - t1
        rr_ratio = reward / max(1e-5, risk)
        
        rationale = (
            f"Bearish trend confirmed. Optimal SELL / SHORT entry zone is between "
            f"{sell_entry_min:,.2f} and {sell_entry_max:,.2f}. "
            f"Set Stop Loss above major resistance level at {sl:,.2f}."
        )
        
    else:
        # Rangebound strategy (Camarilla S3/R3 range)
        s3 = p_lvls.get("S3", spot_price - 1.0 * atr)
        r3 = p_lvls.get("R3", spot_price + 1.0 * atr)
        
        buy_entry_min = s3 - 0.2 * atr
        buy_entry_max = s3 + 0.2 * atr
        sell_entry_min = r3 - 0.2 * atr
        sell_entry_max = r3 + 0.2 * atr
        
        sl = s3 - 1.0 * atr
        t1 = p_lvls.get("P", spot_price)
        t2 = r3
        rr_ratio = 1.5
        rationale = (
            f"Rangebound market. Buy on support bounce near {s3:,.2f} "
            f"or Sell on resistance rejection near {r3:,.2f}. "
            f"Targets are mid-range Pivot P ({t1:,.2f}) and opposite band."
        )

    return {
        "trend_signal": trend_signal,
        "pivot_type": pivot_type,
        "spot_price": spot_price,
        "buy_zone": (buy_entry_min, buy_entry_max),
        "sell_zone": (sell_entry_min, sell_entry_max),
        "stop_loss": sl,
        "target_1": t1,
        "target_2": t2,
        "risk_reward_ratio": rr_ratio,
        "rationale": rationale
    }


# =====================================================================
# 3. MULTI-TIMEFRAME LEVEL SYNTHESIZER
# =====================================================================

def synthesize_multi_timeframe_levels(symbol, fetch_live_data_fn, compute_features_fn=None):
    """
    Fetches market data across 5m, 15m, 1H, 4H, and 1D timeframes,
    computes levels, trend direction, and returns a consolidated dictionary.
    """
    intervals = ["5m", "15m", "1H", "4H", "1D"]
    synthesized = {}
    
    for interval in intervals:
        try:
            df = fetch_live_data_fn(symbol, interval)
            if df.empty or len(df) < 10:
                continue
                
            current_price = float(df["close"].iloc[-1])
            
            # Simple technical bias for this specific timeframe
            # 1. EMA 9 vs EMA 21 crossover
            ema9 = df["close"].ewm(span=9).mean().iloc[-1]
            ema21 = df["close"].ewm(span=21).mean().iloc[-1]
            
            # 2. RSI momentum
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1.0/14.0, min_periods=14, adjust=False).mean().iloc[-1]
            avg_loss = loss.ewm(alpha=1.0/14.0, min_periods=14, adjust=False).mean().iloc[-1]
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs)) if not math.isnan(rs) else 50.0
            
            # Trend determination
            if ema9 > ema21:
                trend = "BUY"
            elif ema9 < ema21:
                trend = "SELL"
            else:
                trend = "HOLD"
                
            # Volatility (ATR simple approximation: high-low rolling average)
            high_low_diff = df["high"] - df["low"]
            atr = float(high_low_diff.rolling(14).mean().iloc[-1])
            if math.isnan(atr) or atr <= 0:
                atr = current_price * 0.01
                
            # Levels
            levels = calculate_pivot_points(df)
            setup = get_trade_setup(current_price, trend, atr, levels)
            
            synthesized[interval] = {
                "price": current_price,
                "trend": trend,
                "rsi": rsi,
                "buy_zone_min": setup["buy_zone"][0],
                "buy_zone_max": setup["buy_zone"][1],
                "sell_zone_min": setup["sell_zone"][0],
                "sell_zone_max": setup["sell_zone"][1],
                "stop_loss": setup["stop_loss"],
                "target_1": setup["target_1"],
                "target_2": setup["target_2"],
                "rr_ratio": setup["risk_reward_ratio"]
            }
        except Exception as e:
            print(f"[Synthesizer] Error in timeframe {interval}: {e}")
            continue
            
    return synthesized
