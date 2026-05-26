package com.manas.engine.indicators;

import java.util.List;

import com.manas.engine.model.Candle;

public class SMAIndicator {

    private final int period;

    public SMAIndicator(int period) {
        this.period = period;
    }

    public double calculate(List<Candle> candles) {
        if (candles.size() < period) return 0;

        double sum = 0;
        for (int i = candles.size() - period; i < candles.size(); i++) {
            sum += candles.get(i).getClose();
        }

        return sum / period;
    }
}