package com.manas.engine.priceaction;

import java.util.List;

import com.manas.engine.model.Candle;

public class BreakoutDetector {

    public boolean bullishBreakout(List<Candle> candles, int lookback) {

        int size = candles.size();

        double prevHigh = Double.MIN_VALUE;

        for (int i = size - lookback - 1; i < size - 1; i++) {
            prevHigh = Math.max(prevHigh, candles.get(i).getHigh());
        }

        return candles.get(size - 1).getHigh() > prevHigh;
    }

    public boolean bearishBreakout(List<Candle> candles, int lookback) {

        int size = candles.size();

        double prevLow = Double.MAX_VALUE;

        for (int i = size - lookback - 1; i < size - 1; i++) {
            prevLow = Math.min(prevLow, candles.get(i).getLow());
        }

        return candles.get(size - 1).getLow() < prevLow;
    }
}