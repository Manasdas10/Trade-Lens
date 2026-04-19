package com.manas.engine.smc;

import java.util.List;

import com.manas.engine.model.Candle;

public class BreakOfStructureDetector {

    public boolean bullishBOS(List<Candle> candles) {

        if (candles.size() < 20) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle swing = candles.get(candles.size() - 10);

        return last.getHigh() > swing.getHigh() &&
               last.getClose() > swing.getHigh();
    }

    public boolean bearishBOS(List<Candle> candles) {

        if (candles.size() < 20) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle swing = candles.get(candles.size() - 10);

        return last.getLow() < swing.getLow() &&
               last.getClose() < swing.getLow();
    }
}