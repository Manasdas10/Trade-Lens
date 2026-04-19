package com.manas.engine.smc;

import com.manas.engine.model.Candle;

public class DisplacementDetector {

    public boolean strongBullish(Candle c){

        double range = c.getHigh() - c.getLow();
        double body = Math.abs(c.getClose() - c.getOpen());

        return body / range > 0.6 && c.getClose() > c.getOpen();
    }

    public boolean strongBearish(Candle c){

        double range = c.getHigh() - c.getLow();
        double body = Math.abs(c.getClose() - c.getOpen());

        return body / range > 0.6 && c.getClose() < c.getOpen();
    }
}