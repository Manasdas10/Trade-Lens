package com.manas.engine.risk;

import com.manas.engine.model.Candle;

public class StopLossManager {

    public double calculateStopLoss(Candle entry){

        return entry.getLow();
    }

    public double calculateTakeProfit(double entryPrice, double stopLoss){

        double risk = entryPrice - stopLoss;

        return entryPrice + risk * 2;
    }
}