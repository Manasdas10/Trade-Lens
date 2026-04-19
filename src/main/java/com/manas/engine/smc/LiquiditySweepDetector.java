package com.manas.engine.smc;

import java.util.List;

import com.manas.engine.model.Candle;

public class LiquiditySweepDetector {

    public boolean sweepHigh(List<Candle> candles){

        int size = candles.size();

        Candle current = candles.get(size-1);
        Candle prev = candles.get(size-2);

        return current.getHigh() > prev.getHigh()
                && current.getClose() < current.getHigh();
    }

    public boolean sweepLow(List<Candle> candles){

        int size = candles.size();

        Candle current = candles.get(size-1);
        Candle prev = candles.get(size-2);

        return current.getLow() < prev.getLow()
                && current.getClose() > current.getLow();
    }
}