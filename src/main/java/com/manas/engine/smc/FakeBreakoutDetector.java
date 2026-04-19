package com.manas.engine.smc;

import java.util.List;

import com.manas.engine.model.Candle;

public class FakeBreakoutDetector {

    public boolean isBullTrap(List<Candle> candles) {

        if (candles.size() < 5) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle prev = candles.get(candles.size() - 2);

        return last.getHigh() > prev.getHigh() &&
               last.getClose() < last.getOpen();
    }

    public boolean isBearTrap(List<Candle> candles) {

        if (candles.size() < 5) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle prev = candles.get(candles.size() - 2);

        return last.getLow() < prev.getLow() &&
               last.getClose() > last.getOpen();
    }
}