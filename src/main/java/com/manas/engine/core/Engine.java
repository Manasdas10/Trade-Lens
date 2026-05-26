package com.manas.engine.core;

import java.util.List;

import com.manas.engine.model.Candle;
import com.manas.engine.strategy.Signal;
import com.manas.engine.strategy.Strategy;

public class Engine {

    private final Strategy strategy;

    public Engine(Strategy strategy) {
        this.strategy = strategy;
    }

    public void run(List<Candle> candles) {

        System.out.println("=================================");
        System.out.println("RUNNING STRATEGY BACKTEST");
        System.out.println("=================================");

        int buyCount = 0;
        int sellCount = 0;
        int holdCount = 0;

        for (int i = 120; i < candles.size(); i++) {

            List<Candle> subCandles =
                    candles.subList(0, i + 1);

            Signal signal =
                    strategy.generateSignal(subCandles);

            Candle current =
                    candles.get(i);

           System.out.println(
                "Index: "
                + i
                + " -> "
                + signal
                + " | Close: "
                + current.getClose()
            );

            switch (signal) {
                case BUY:
                    buyCount++;
                    break;

                case SELL:
                    sellCount++;
                    break;

                default:
                    holdCount++;
            }
        }

        System.out.println("=================================");
        System.out.println("BACKTEST COMPLETE");
        System.out.println("BUY SIGNALS : " + buyCount);
        System.out.println("SELL SIGNALS: " + sellCount);
        System.out.println("HOLD SIGNALS: " + holdCount);
        System.out.println("=================================");
    }
}