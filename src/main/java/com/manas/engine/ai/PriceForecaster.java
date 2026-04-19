package com.manas.engine.ai;

import java.util.List;

import com.manas.engine.model.Candle;

public class PriceForecaster {

    public PredictionResult forecast(List<Candle> candles){

        int n = candles.size();

        double last = candles.get(n-1).getClose();
        double prev = candles.get(n-2).getClose();

        double momentum = last - prev;

        double prediction = last + momentum;

        double confidence = Math.min(Math.abs(momentum) / last, 1.0);

        return new PredictionResult(prediction, confidence);
    }
}