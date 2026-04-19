package com.manas.engine.strategy;

import java.util.List;

import com.manas.engine.model.Candle;
import com.manas.engine.priceaction.BreakoutDetector;

public class BreakoutStrategy implements Strategy {

    private final BreakoutDetector breakout = new BreakoutDetector();

    @Override
    public Signal generateSignal(List<Candle> candles) {

        if (candles.size() < 20)
            return Signal.HOLD;

        if (breakout.bullishBreakout(candles, 10))
            return Signal.BUY;

        if (breakout.bearishBreakout(candles, 10))
            return Signal.SELL;

        return Signal.HOLD;
    }
}