package com.manas.engine.ai;

import java.util.ArrayList;
import java.util.List;

import com.manas.engine.model.Candle;

public class PriceFeatureBuilder {

    public static List<double[]> buildFeatures(List<Candle> candles){

        List<double[]> features = new ArrayList<>();

        for(int i = 1; i < candles.size(); i++){

            Candle c = candles.get(i);
            Candle prev = candles.get(i-1);

            double returnPct = (c.getClose() - prev.getClose()) / prev.getClose();

            double body = Math.abs(c.getClose() - c.getOpen());
            double range = c.getHigh() - c.getLow();

            double volume = c.getVolume();

            features.add(new double[]{
                    c.getClose(),
                    returnPct,
                    body,
                    range,
                    volume
            });
        }

        return features;
    }
}