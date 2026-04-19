package com.manas.engine.priceaction;

import java.util.List;

import com.manas.engine.model.Candle;

public class ImpulseCandleDetector {

    public boolean isBullishImpulse(List<Candle> candles) {

        Candle last = candles.get(candles.size() - 1);

        double body = Math.abs(last.getClose() - last.getOpen());
        double range = last.getHigh() - last.getLow();

        return last.getClose() > last.getOpen() &&
               body > range * 0.6;
    }

    public boolean isBearishImpulse(List<Candle> candles) {

        Candle last = candles.get(candles.size() - 1);

        double body = Math.abs(last.getClose() - last.getOpen());
        double range = last.getHigh() - last.getLow();

        return last.getClose() < last.getOpen() &&
               body > range * 0.6;
    }
}