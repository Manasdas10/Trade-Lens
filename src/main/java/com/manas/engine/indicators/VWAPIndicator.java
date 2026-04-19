package com.manas.engine.indicators;

import java.util.List;

import com.manas.engine.model.Candle;

public class VWAPIndicator {

    public double calculate(List<Candle> candles) {

        double cumulativePV = 0.0;
        double cumulativeVolume = 0.0;

        for (Candle candle : candles) {

            double typicalPrice = 
                    (candle.getHigh() + candle.getLow() + candle.getClose()) / 3.0;

            cumulativePV += typicalPrice * candle.getVolume();
            cumulativeVolume += candle.getVolume();
        }

        return cumulativePV / cumulativeVolume;
    }
}