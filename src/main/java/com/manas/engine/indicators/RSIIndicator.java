package com.manas.engine.indicators;

import com.manas.engine.model.Candle;

import java.util.List;

public class RSIIndicator implements Indicator {

    private final int period;

    public RSIIndicator(int period) {
        this.period = period;
    }

    @Override
    public double calculate(List<Candle> candles) {

        if (candles == null || candles.size() <= period) {
            throw new IllegalArgumentException("Not enough candles to calculate RSI");
        }

        double gain = 0.0;
        double loss = 0.0;

        for (int i = candles.size() - period; i < candles.size(); i++) {
            double change = candles.get(i).getClose() - candles.get(i - 1).getClose();

            if (change > 0) {
                gain += change;
            } else {
                loss += Math.abs(change);
            }
        }

        if (loss == 0) {
            return 100;
        }

        double rs = gain / loss;
        return 100 - (100 / (1 + rs));
    }
}