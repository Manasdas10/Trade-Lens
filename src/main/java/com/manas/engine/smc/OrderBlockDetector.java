package com.manas.engine.smc;

import java.util.List;
import com.manas.engine.model.Candle;

public class OrderBlockDetector {

    public boolean isBullishOrderBlock(List<Candle> candles) {

        if (candles.size() < 5) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle prev = candles.get(candles.size() - 2);

        return prev.getClose() < prev.getOpen() &&  // bearish candle
               last.getClose() > prev.getHigh();   // strong break
    }

    public boolean isBearishOrderBlock(List<Candle> candles) {

        if (candles.size() < 5) return false;

        Candle last = candles.get(candles.size() - 1);
        Candle prev = candles.get(candles.size() - 2);

        return prev.getClose() > prev.getOpen() &&  // bullish candle
               last.getClose() < prev.getLow();    // strong break
    }
}