package com.manas.engine.priceaction;

import java.util.List;

import com.manas.engine.model.Candle;

public class TrendStructure {

    public boolean bullishTrend(List<Candle> candles){

        int size = candles.size();

        double lastHigh = candles.get(size-1).getHigh();
        double prevHigh = candles.get(size-2).getHigh();

        double lastLow = candles.get(size-1).getLow();
        double prevLow = candles.get(size-2).getLow();

        return lastHigh > prevHigh && lastLow > prevLow;
    }

    public boolean bearishTrend(List<Candle> candles){

        int size = candles.size();

        double lastHigh = candles.get(size-1).getHigh();
        double prevHigh = candles.get(size-2).getHigh();

        double lastLow = candles.get(size-1).getLow();
        double prevLow = candles.get(size-2).getLow();

        return lastHigh < prevHigh && lastLow < prevLow;
    }
}