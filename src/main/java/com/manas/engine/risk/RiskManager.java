package com.manas.engine.risk;

import java.util.List;

import com.manas.engine.indicators.ATRIndicator;
import com.manas.engine.model.Candle;
import com.manas.engine.model.Position;

public class RiskManager {

    private final double accountBalance;
    private final double riskPerTradePercent;
    private final ATRIndicator atrIndicator;

    public RiskManager(double accountBalance, double riskPerTradePercent, int atrPeriod) {
        this.accountBalance = accountBalance;
        this.riskPerTradePercent = riskPerTradePercent;
        this.atrIndicator = new ATRIndicator(atrPeriod);
    }

    public Position createPosition(List<Candle> candles, double entryPrice) {

        double atr = atrIndicator.calculate(candles);

        double stopLoss = entryPrice - (atr * 1.5);

        double riskAmount = accountBalance * riskPerTradePercent;

        double riskPerUnit = entryPrice - stopLoss;

        int quantity = (int) (riskAmount / riskPerUnit);

        if (quantity <= 0) {
            quantity = 1;
        }

        return new Position(entryPrice, stopLoss, quantity);
    }
}