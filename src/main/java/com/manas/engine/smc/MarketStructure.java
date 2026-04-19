package com.manas.engine.smc;

import java.util.List;

import com.manas.engine.model.Candle;

public class MarketStructure {

    public enum Trend {
        BULLISH,
        BEARISH,
        SIDEWAYS
    }

    public Trend detectTrend(List<Candle> candles) {

        if (candles.size() < 20) return Trend.SIDEWAYS;

        Candle last = candles.get(candles.size() - 1);
        Candle prev = candles.get(candles.size() - 5);

        if (last.getHigh() > prev.getHigh() &&
            last.getLow() > prev.getLow()) {

            return Trend.BULLISH;
        }

        if (last.getHigh() < prev.getHigh() &&
            last.getLow() < prev.getLow()) {

            return Trend.BEARISH;
        }

        return Trend.SIDEWAYS;
    }
}