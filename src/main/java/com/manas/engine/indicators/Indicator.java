package com.manas.engine.indicators;

import java.util.List;

import com.manas.engine.model.Candle;

public interface Indicator {
    double calculate(List<Candle> candles);
}