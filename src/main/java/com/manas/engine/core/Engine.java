package com.manas.engine.core;

import java.util.List;

import com.manas.engine.ai.PredictionResult;
import com.manas.engine.ai.PriceForecaster;
import com.manas.engine.backtest.BacktestEngine;
import com.manas.engine.data.CSVLoader;
import com.manas.engine.model.Candle;
import com.manas.engine.strategy.EMACrossoverStrategy;
import com.manas.engine.strategy.Strategy;

public class Engine {

    public static void main(String[] args) {

        System.out.println("=================================");
        System.out.println("       MARKET ENGINE STARTED");
        System.out.println("=================================");

        try {

            // Load historical data
            List<Candle> candles = CSVLoader.load("src/main/resources/nifty.csv");

            System.out.println("Candles Loaded: " + candles.size());

            if (candles.isEmpty()) {
                System.out.println("ERROR: No candle data found.");
                return;
            }

            // -----------------------------
            // AI FORECAST MODULE
            // -----------------------------

            PriceForecaster ai = new PriceForecaster();

            PredictionResult prediction = ai.forecast(candles);

            System.out.println("\n===== AI FORECAST =====");
            System.out.println("Predicted Price: " + prediction.getPredictedPrice());
            System.out.println("Confidence: " + prediction.getConfidence());
            System.out.println("=======================\n");

            // -----------------------------
            // STRATEGY SELECTION
            // -----------------------------

            Strategy strategy = new EMACrossoverStrategy();

            System.out.println("Strategy Selected: " + strategy.getClass().getSimpleName());

            // -----------------------------
            // BACKTEST
            // -----------------------------

            BacktestEngine engine = new BacktestEngine(strategy);

            System.out.println("\nStarting Backtest...\n");

            engine.run(candles);

            System.out.println("\nBacktest Completed Successfully.");

        } catch (Exception e) {

            System.out.println("ENGINE ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("=================================");
        System.out.println("       MARKET ENGINE FINISHED");
        System.out.println("=================================");
    }
}