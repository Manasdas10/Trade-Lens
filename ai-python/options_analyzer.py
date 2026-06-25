"""
options_analyzer.py — Futures & Options (F&O) Analytics Engine
Computes support/resistance pivot levels, strike prices, option Greeks (Black-Scholes),
and generates options strategy recommendations.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm

# =====================================================================
# 1. PIVOT POINTS CALCULATOR (SUPPORT / RESISTANCE)
# =====================================================================

def calculate_pivot_points(df):
    """
    Computes Classic, Fibonacci, and Camarilla Pivot Points.
    Uses the latest completed bar (row -1 or -2 depending on live status)
    to calculate levels for the current/upcoming bar.
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
    r4_cam = close + (range_val * 1.1 / 2.0)  # Strong breakout level
    s4_cam = close - (range_val * 1.1 / 2.0)  # Strong breakdown level

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
# 2. STRIKE PRICE STEP DETECTOR
# =====================================================================

def detect_strike_step(symbol, spot_price):
    """
    Returns the standard options strike price step interval based on symbol or spot price.
    """
    sym = symbol.upper()
    
    # Specific popular Indian Indices and Stocks
    if "NSEI" in sym or "NIFTY" in sym:
        return 50.0  # NIFTY 50 strike interval is 50
    elif "BANKNIFTY" in sym or "NSEBANK" in sym:
        return 100.0  # BANK NIFTY strike interval is 100
    elif "CNXFINANCE" in sym or "FINNIFTY" in sym:
        return 50.0  # FIN NIFTY strike interval is 50
    
    # Cryptocurrencies
    if "BTC" in sym:
        return 500.0 if spot_price < 50000 else 1000.0
    elif "ETH" in sym:
        return 50.0 if spot_price < 2000 else 100.0
    
    # Generic rules based on spot price level
    if spot_price > 20000:
        return 100.0
    elif spot_price > 10000:
        return 50.0
    elif spot_price > 2000:
        return 20.0
    elif spot_price > 1000:
        return 10.0
    elif spot_price > 500:
        return 5.0
    elif spot_price > 100:
        return 2.5
    elif spot_price > 20:
        return 1.0
    else:
        return 0.5


def get_recommended_strikes(spot_price, strike_step, num_strikes=5):
    """
    Returns a dictionary of Call and Put strikes (ATM, ITM, OTM).
    """
    # ATM is the strike closest to the current spot price
    atm = round(spot_price / strike_step) * strike_step
    
    itm_calls = [atm - (i * strike_step) for i in range(1, num_strikes + 1)]
    otm_calls = [atm + (i * strike_step) for i in range(1, num_strikes + 1)]
    
    itm_puts = [atm + (i * strike_step) for i in range(1, num_strikes + 1)]
    otm_puts = [atm - (i * strike_step) for i in range(1, num_strikes + 1)]
    
    return {
        "ATM": atm,
        "ITM_Calls": sorted(itm_calls),
        "OTM_Calls": sorted(otm_calls),
        "ITM_Puts": sorted(itm_puts),
        "OTM_Puts": sorted(otm_puts)
    }


# =====================================================================
# 3. BLACK-SCHOLES OPTIONS GREEKS CALCULATOR
# =====================================================================

def calculate_black_scholes_greeks(spot, strike, t_days, r, iv, option_type="CE"):
    """
    Calculates theoretical option premium and Greeks (Delta, Gamma, Theta, Vega)
    using the Black-Scholes-Merton model.
    
    Parameters
    ----------
    spot : float
        Current spot price of asset.
    strike : float
        Option strike price.
    t_days : float
        Days to expiration.
    r : float
        Risk-free interest rate (annualized, e.g. 0.065 for 6.5%).
    iv : float
        Implied Volatility (annualized, e.g. 0.15 for 15%).
    option_type : str
        "CE" (Call) or "PE" (Put).
        
    Returns
    -------
    dict
        {price, delta, gamma, theta, vega}
    """
    # Expiry in years
    T = max(1e-5, t_days / 365.0)
    # Ensure inputs are valid
    spot = max(1e-5, spot)
    strike = max(1e-5, strike)
    iv = max(1e-4, iv)
    
    # Calculate d1 and d2
    d1 = (math.log(spot / strike) + (r + (iv ** 2) / 2.0) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    
    # Normal distribution stats
    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    n_minus_d1 = norm.cdf(-d1)
    n_minus_d2 = norm.cdf(-d2)
    
    # PDF of d1 for Gamma/Vega
    pdf_d1 = norm.pdf(d1)
    
    # Calculations
    if option_type.upper() == "CE":
        # Call Option
        price = spot * n_d1 - strike * math.exp(-r * T) * n_d2
        delta = n_d1
        # Theta annualized, then divided by 365 for daily decay
        theta_annual = -(spot * pdf_d1 * iv) / (2 * math.sqrt(T)) - r * strike * math.exp(-r * T) * n_d2
        theta = theta_annual / 365.0
    else:
        # Put Option
        price = strike * math.exp(-r * T) * n_minus_d2 - spot * n_minus_d1
        delta = n_minus_d1 - 1.0
        theta_annual = -(spot * pdf_d1 * iv) / (2 * math.sqrt(T)) + r * strike * math.exp(-r * T) * n_minus_d2
        theta = theta_annual / 365.0
        
    gamma = pdf_d1 / (spot * iv * math.sqrt(T))
    # Vega is derivative w.r.t volatility. Divided by 100 to show price change per 1% change in IV
    vega = (spot * math.sqrt(T) * pdf_d1) / 100.0
    
    # Clamp price to intrinsic value if numerical error occurs
    intrinsic = max(0.0, spot - strike if option_type.upper() == "CE" else strike - spot)
    price = max(price, intrinsic)
    
    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }


# =====================================================================
# 4. OPTION CHAIN GENERATOR
# =====================================================================

def generate_options_chain(spot, iv, strike_step, days_to_expiry, risk_free_rate, num_strikes=11):
    """
    Generates a structured DataFrame representing a double-sided Options Chain
    centered around the spot price.
    """
    recommended = get_recommended_strikes(spot, strike_step, num_strikes=num_strikes//2)
    atm = recommended["ATM"]
    
    # Combine strikes into a sorted list
    strikes = sorted(recommended["ITM_Calls"] + [atm] + recommended["OTM_Calls"])
    
    rows = []
    for K in strikes:
        # Call pricing & Greeks
        c_stats = calculate_black_scholes_greeks(spot, K, days_to_expiry, risk_free_rate, iv, "CE")
        # Put pricing & Greeks
        p_stats = calculate_black_scholes_greeks(spot, K, days_to_expiry, risk_free_rate, iv, "PE")
        
        # Determine status
        c_itm = "ITM" if spot > K else "OTM" if spot < K else "ATM"
        p_itm = "ITM" if spot < K else "OTM" if spot > K else "ATM"
        
        rows.append({
            "c_delta": c_stats["delta"],
            "c_gamma": c_stats["gamma"],
            "c_theta": c_stats["theta"],
            "c_vega": c_stats["vega"],
            "c_price": c_stats["price"],
            "c_status": c_itm,
            "strike": K,
            "p_price": p_stats["price"],
            "p_status": p_itm,
            "p_delta": p_stats["delta"],
            "p_gamma": p_stats["gamma"],
            "p_theta": p_stats["theta"],
            "p_vega": p_stats["vega"],
        })
        
    return pd.DataFrame(rows)


# =====================================================================
# 5. ESTIMATE VOLATILITY FROM OHLCV HISTORICAL DATA
# =====================================================================

def estimate_historical_volatility(df, window=20):
    """
    Calculates historical volatility of log returns as a proxy for Implied Volatility.
    Returns annualized volatility decimal (e.g. 0.18 for 18% volatility).
    """
    if df.empty or len(df) < window:
        return 0.15  # Fallback to standard 15% IV
    
    close = df["close"].values
    log_returns = np.log(close[1:] / close[:-1])
    
    # Rolling standard deviation of returns
    std = np.std(log_returns[-window:])
    
    # Annualize standard deviation (assume 252 trading days in a year)
    annualized_vol = std * np.sqrt(252.0)
    
    return max(0.05, float(annualized_vol))  # Minimum 5% volatility


# =====================================================================
# 6. F&O STRATEGY RECOMMENDATION GENERATOR
# =====================================================================

def get_fo_recommendations(symbol, spot_price, trend_signal, atr, levels, days_to_expiry=7, r=0.065, iv=0.15):
    """
    Generates actionable trade recommendations for Futures and Options.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol.
    spot_price : float
        Current asset price.
    trend_signal : str
        "BUY", "SELL", or "HOLD".
    atr : float
        Average True Range (volatility measure).
    levels : dict
        Pivot points from calculate_pivot_points().
    """
    strike_step = detect_strike_step(symbol, spot_price)
    strikes = get_recommended_strikes(spot_price, strike_step, num_strikes=3)
    
    atm = strikes["ATM"]
    itm_ce = strikes["ITM_Calls"][-1]  # Closest ITM Call (e.g., Spot - step)
    otm_ce = strikes["OTM_Calls"][0]   # Closest OTM Call (e.g., Spot + step)
    itm_pe = strikes["ITM_Puts"][0]    # Closest ITM Put (e.g., Spot + step)
    otm_pe = strikes["OTM_Puts"][-1]   # Closest OTM Put (e.g., Spot - step)

    # Use Camarilla pivot levels if available, else Classic
    pivot_type = "camarilla" if "camarilla" in levels else "classic"
    p_lvls = levels[pivot_type]
    
    rec = {
        "pivot_type": pivot_type,
        "futures": {},
        "option_buying": {},
        "option_selling": {},
        "option_spread": {}
    }

    # 1. FUTURES RECOMMENDATION
    if trend_signal == "BUY":
        entry = spot_price
        # Set stop loss below support S1 or 1.5 ATR
        sl = max(p_lvls.get("S1", spot_price - 1.5 * atr), spot_price - 1.5 * atr)
        # Set target near resistance R2 or 2 ATR
        target = min(p_lvls.get("R2", spot_price + 2 * atr), spot_price + 2 * atr)
        rec["futures"] = {
            "action": "BUY / LONG",
            "entry": entry,
            "sl": sl,
            "target": target,
            "rr_ratio": (target - entry) / max(1e-5, entry - sl),
            "rationale": f"Trend is bullish. Entry at spot, stop-loss below {pivot_type.capitalize()} S1/ATR support."
        }
    elif trend_signal == "SELL":
        entry = spot_price
        sl = min(p_lvls.get("R1", spot_price + 1.5 * atr), spot_price + 1.5 * atr)
        target = max(p_lvls.get("S2", spot_price - 2 * atr), spot_price - 2 * atr)
        rec["futures"] = {
            "action": "SELL / SHORT",
            "entry": entry,
            "sl": sl,
            "target": target,
            "rr_ratio": (entry - target) / max(1e-5, sl - entry),
            "rationale": f"Trend is bearish. Short at spot, stop-loss above {pivot_type.capitalize()} R1/ATR resistance."
        }
    else:
        rec["futures"] = {
            "action": "NO TRADE (HOLD)",
            "entry": spot_price,
            "sl": spot_price - atr,
            "target": spot_price + atr,
            "rr_ratio": 1.0,
            "rationale": "Market is rangebound. Wait for breakout of Pivot levels before initiating Futures trade."
        }

    # Helper function to get pricing & Greeks
    def get_opt_info(strike, opt_type):
        greeks = calculate_black_scholes_greeks(spot_price, strike, days_to_expiry, r, iv, opt_type)
        return {
            "strike": strike,
            "premium": greeks["price"],
            "delta": greeks["delta"],
            "theta": greeks["theta"],
            "vega": greeks["vega"]
        }

    # 2. OPTIONS BUYING RECOMMENDATION
    if trend_signal == "BUY":
        # Buy ATM or close ITM Call
        strike_to_buy = atm
        opt_info = get_opt_info(strike_to_buy, "CE")
        
        # Stop loss at 40% premium decay or price below S1 support
        sl_premium = opt_info["premium"] * 0.6
        # Target at 60% premium gain
        target_premium = opt_info["premium"] * 1.6
        
        rec["option_buying"] = {
            "action": f"BUY Call (CE)",
            "strike": strike_to_buy,
            "premium": opt_info["premium"],
            "sl": sl_premium,
            "target": target_premium,
            "delta": opt_info["delta"],
            "theta": opt_info["theta"],
            "rationale": f"Buy At-The-Money CE strike {strike_to_buy}. High Delta ({opt_info['delta']:.2f}) will capture spot movement."
        }
    elif trend_signal == "SELL":
        strike_to_buy = atm
        opt_info = get_opt_info(strike_to_buy, "PE")
        sl_premium = opt_info["premium"] * 0.6
        target_premium = opt_info["premium"] * 1.6
        
        rec["option_buying"] = {
            "action": f"BUY Put (PE)",
            "strike": strike_to_buy,
            "premium": opt_info["premium"],
            "sl": sl_premium,
            "target": target_premium,
            "delta": opt_info["delta"],
            "theta": opt_info["theta"],
            "rationale": f"Buy At-The-Money PE strike {strike_to_buy}. Spot decline will raise option value rapidly."
        }
    else:
        rec["option_buying"] = {
            "action": "NO BUY TRADE",
            "strike": atm,
            "premium": 0,
            "sl": 0,
            "target": 0,
            "delta": 0,
            "theta": 0,
            "rationale": "High Theta decay risk during consolidation. Avoid buying options in neutral market."
        }

    # 3. OPTIONS SELLING RECOMMENDATION (WRITING - HEDGED/LIMIT RISK EXPECTED)
    if trend_signal == "BUY":
        # Sell Out-of-the-Money Put (PE) to collect premium
        strike_to_sell = otm_pe
        opt_info = get_opt_info(strike_to_sell, "PE")
        
        rec["option_selling"] = {
            "action": f"SELL Put (PE) - Writer",
            "strike": strike_to_sell,
            "premium": opt_info["premium"],
            "sl": opt_info["premium"] * 2.0,  # Stop Loss at double premium
            "target": opt_info["premium"] * 0.1,  # Target near 90% decay
            "delta": opt_info["delta"],
            "theta": opt_info["theta"],
            "rationale": f"Sell Out-of-the-Money PE strike {strike_to_sell}. Earn premium decay as long as price stays above S1 level."
        }
    elif trend_signal == "SELL":
        # Sell Out-of-the-Money Call (CE)
        strike_to_sell = otm_ce
        opt_info = get_opt_info(strike_to_sell, "CE")
        
        rec["option_selling"] = {
            "action": f"SELL Call (CE) - Writer",
            "strike": strike_to_sell,
            "premium": opt_info["premium"],
            "sl": opt_info["premium"] * 2.0,
            "target": opt_info["premium"] * 0.1,
            "delta": opt_info["delta"],
            "theta": opt_info["theta"],
            "rationale": f"Sell Out-of-the-Money CE strike {strike_to_sell}. Profit from time decay as price remains capped by R1 resistance."
        }
    else:
        # Neutral market: Sell both OTM Call and OTM Put (Short Strangle)
        ce_info = get_opt_info(otm_ce, "CE")
        pe_info = get_opt_info(otm_pe, "PE")
        
        rec["option_selling"] = {
            "action": "SELL Strangle (CE & PE)",
            "strike": f"{otm_pe} PE & {otm_ce} CE",
            "premium": ce_info["premium"] + pe_info["premium"],
            "sl": (ce_info["premium"] + pe_info["premium"]) * 1.5,
            "target": (ce_info["premium"] + pe_info["premium"]) * 0.2,
            "delta": ce_info["delta"] + pe_info["delta"],
            "theta": ce_info["theta"] + pe_info["theta"],
            "rationale": f"Consolidating market. Sell CE at {otm_ce} and PE at {otm_pe} to double-harvest Theta time decay."
        }

    # 4. OPTION SPREAD RECOMMENDATION (DEFINED RISK)
    if trend_signal == "BUY":
        # Bull Call Spread: Buy ITM CE, Sell OTM CE
        buy_ce = get_opt_info(itm_ce, "CE")
        sell_ce = get_opt_info(otm_ce, "CE")
        net_debit = buy_ce["premium"] - sell_ce["premium"]
        max_profit = (otm_ce - itm_ce) - net_debit
        
        rec["option_spread"] = {
            "strategy": "Bull Call Spread",
            "legs": [
                f"BUY CE strike {itm_ce} (Cost: {buy_ce['premium']:.2f})",
                f"SELL CE strike {otm_ce} (Credit: {sell_ce['premium']:.2f})"
            ],
            "net_cost": net_debit,
            "max_risk": net_debit,
            "max_profit": max_profit,
            "rr_ratio": max_profit / max(1e-5, net_debit),
            "rationale": "Bullish spread. Buying ITM Call while offset-selling OTM Call to decrease net cost & hedge IV drops."
        }
    elif trend_signal == "SELL":
        # Bear Put Spread: Buy ITM PE, Sell OTM PE
        buy_pe = get_opt_info(itm_pe, "PE")
        sell_pe = get_opt_info(otm_pe, "PE")
        net_debit = buy_pe["premium"] - sell_pe["premium"]
        max_profit = (itm_pe - otm_pe) - net_debit
        
        rec["option_spread"] = {
            "strategy": "Bear Put Spread",
            "legs": [
                f"BUY PE strike {itm_pe} (Cost: {buy_pe['premium']:.2f})",
                f"SELL PE strike {otm_pe} (Credit: {sell_pe['premium']:.2f})"
            ],
            "net_cost": net_debit,
            "max_risk": net_debit,
            "max_profit": max_profit,
            "rr_ratio": max_profit / max(1e-5, net_debit),
            "rationale": "Bearish spread. Buying ITM Put and selling OTM Put to limit maximum risk while targeting Pivot S2."
        }
    else:
        # Iron Condor (Neutral): Sell OTM PE, Buy deeper OTM PE, Sell OTM CE, Buy deeper OTM CE
        wider_otm_pe = strikes["OTM_Puts"][0]  # Even lower put
        wider_otm_ce = strikes["OTM_Calls"][-1] # Even higher call
        
        s_ce = get_opt_info(otm_ce, "CE")
        b_ce = get_opt_info(wider_otm_ce, "CE")
        s_pe = get_opt_info(otm_pe, "PE")
        b_pe = get_opt_info(wider_otm_pe, "PE")
        
        net_credit = (s_ce["premium"] - b_ce["premium"]) + (s_pe["premium"] - b_pe["premium"])
        max_risk = (otm_ce - otm_pe) - net_credit # Simplified risk based on spread width
        
        rec["option_spread"] = {
            "strategy": "Iron Condor",
            "legs": [
                f"BUY PE strike {wider_otm_pe} (Cost: {b_pe['premium']:.2f})",
                f"SELL PE strike {otm_pe} (Credit: {s_pe['premium']:.2f})",
                f"SELL CE strike {otm_ce} (Credit: {s_ce['premium']:.2f})",
                f"BUY CE strike {wider_otm_ce} (Cost: {b_ce['premium']:.2f})"
            ],
            "net_cost": -net_credit, # Negative net cost = net credit
            "max_risk": max_risk,
            "max_profit": net_credit,
            "rr_ratio": net_credit / max(1e-5, max_risk),
            "rationale": "Rangebound protection. Iron Condor generates a net credit with capped risk if asset remains inside S1-R1 boundaries."
        }

    return rec
