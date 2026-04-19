package com.manas.engine.strategy;

import java.util.List;

import com.manas.engine.model.Candle;

public interface Strategy {
    Signal generateSignal(List<Candle> candles);
}