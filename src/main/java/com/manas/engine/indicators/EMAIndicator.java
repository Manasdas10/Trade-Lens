package com.manas.engine.indicators;

import java.util.List;

import com.manas.engine.model.Candle;

public class EMAIndicator implements Indicator {

    private final int period;

    public EMAIndicator(int period) {
        this.period = period;
    }

    @Override
    public double calculate(List<Candle> candles) {
        if (candles == null || candles.size() < period) {
            throw new IllegalArgumentException("Not enough candles to calculate EMA");
        }

        double multiplier = 2.0 / (period + 1);

        // Start with SMA for first EMA value
        double sma = 0.0;
        for (int i = 0; i < period; i++) {
            sma += candles.get(i).getClose();
        }
        sma /= period;

        double ema = sma;

        for (int i = period; i < candles.size(); i++) {
            double close = candles.get(i).getClose();
            ema = ((close - ema) * multiplier) + ema;
        }

        return ema;
    }
}