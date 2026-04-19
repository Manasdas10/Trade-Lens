package com.manas.engine.indicators;

import java.util.List;

import com.manas.engine.model.Candle;

public class ATRIndicator implements Indicator {

    private final int period;

    public ATRIndicator(int period) {
        this.period = period;
    }

    @Override
    public double calculate(List<Candle> candles) {

        if (candles == null || candles.size() <= period) {
            throw new IllegalArgumentException("Not enough candles to calculate ATR");
        }

        double atr = 0.0;

        for (int i = candles.size() - period; i < candles.size(); i++) {

            Candle current = candles.get(i);
            Candle previous = candles.get(i - 1);

            double highLow = current.getHigh() - current.getLow();
            double highClose = Math.abs(current.getHigh() - previous.getClose());
            double lowClose = Math.abs(current.getLow() - previous.getClose());

            double trueRange = Math.max(highLow, Math.max(highClose, lowClose));

            atr += trueRange;
        }

        return atr / period;
    }
}