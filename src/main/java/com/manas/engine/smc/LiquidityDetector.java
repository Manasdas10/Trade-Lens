package com.manas.engine.smc;

import java.util.List;

import com.manas.engine.model.Candle;

public class LiquidityDetector {

    public boolean bullishLiquiditySweep(List<Candle> candles) {

        if (candles.size() < 10) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle prevLow = candles.get(candles.size() - 5);

        return last.getLow() < prevLow.getLow() &&
               last.getClose() > prevLow.getLow();
    }

    public boolean bearishLiquiditySweep(List<Candle> candles) {

        if (candles.size() < 10) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle prevHigh = candles.get(candles.size() - 5);

        return last.getHigh() > prevHigh.getHigh() &&
               last.getClose() < prevHigh.getHigh();
    }
}