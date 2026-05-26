package com.manas.engine;

import java.util.List;

import com.manas.engine.ai.AIPredictor;
import com.manas.engine.analytics.MonteCarloSimulator;
import com.manas.engine.analytics.PerformanceAnalyzer;
import com.manas.engine.core.Engine;
import com.manas.engine.data.YahooDataFetcher;
import com.manas.engine.model.Candle;
import com.manas.engine.strategy.EMACrossoverStrategy;
import com.manas.engine.strategy.Strategy;

import yahoofinance.histquotes.Interval;

public class Main {

    public static void main(String[] args) {

        try {

            System.out.println("=================================");
            System.out.println(" MARKET ENGINE STARTED ");
            System.out.println("=================================");

            // FETCH LIVE MARKET DATA
            List<Candle> candles =
                    YahooDataFetcher.fetch(
                            "^NSEI",
                            Interval.DAILY
                    );

            System.out.println(
                    "Candles Loaded: "
                    + candles.size()
            );

            // STRATEGY
            Strategy strategy =
                    new EMACrossoverStrategy();

            // RUN ENGINE
            Engine engine =
                    new Engine(strategy);

            engine.run(candles);

            // GET LATEST CLOSE PRICE
            double latestClose =
                    candles.get(candles.size() - 1)
                           .getClose();

            // AI PREDICTION
            String aiSignal =
                    AIPredictor.runPrediction(
                            latestClose
                    );

            System.out.println(
                    "AI Signal: " + aiSignal
            );

            // PERFORMANCE ANALYSIS
            PerformanceAnalyzer analyzer =
                    new PerformanceAnalyzer();

            analyzer.analyze();

            // MONTE CARLO SIMULATION
            MonteCarloSimulator simulator =
                    new MonteCarloSimulator();

            simulator.runSimulation();

            System.out.println("=================================");
            System.out.println(" BACKTEST COMPLETE ");
            System.out.println("=================================");

        } catch (Exception e) {

            System.out.println(
                    "MAIN ERROR: "
                    + e.getMessage()
            );

            e.printStackTrace();
        }
    }
}