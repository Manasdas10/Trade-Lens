package com.manas.engine.backtest;

import java.util.List;

import com.manas.engine.indicators.ATRIndicator;
import com.manas.engine.model.Candle;
import com.manas.engine.strategy.Signal;
import com.manas.engine.strategy.Strategy;

public class BacktestEngine {

    private final Strategy strategy;
    private final ATRIndicator atr;

    private int wins = 0;
    private int losses = 0;
    private double netProfit = 0.0;
    private double maxDrawdown = 0.0;
    private double equity = 0.0;
    private double peakEquity = 0.0;

    public BacktestEngine(Strategy strategy) {
        this.strategy = strategy;
        this.atr = new ATRIndicator(14);
    }

    public void run(List<Candle> candles) {

        for (int i = 50; i < candles.size() - 1; i++) {

            // Progress indicator
            if (i % 10000 == 0) {
                System.out.println("Processed candles: " + i);
            }

            List<Candle> subList = candles.subList(0, i + 1);
            Signal signal = strategy.generateSignal(subList);

            if (signal == Signal.HOLD) continue;

            double atrValue = atr.calculate(subList);
            double entryPrice = candles.get(i).getClose();

            // Volatility filter
            double atrPercent = (atrValue / entryPrice) * 100;

            if (atrPercent < 0.4) continue;

            double stopLoss;
            double target;

            if (signal == Signal.BUY) {

                stopLoss = entryPrice - (atrValue * 1.0);
                target = entryPrice + (atrValue * 2.0);

                executeTrade(true, entryPrice, stopLoss, target, candles, i);
            }

            if (signal == Signal.SELL) {

                stopLoss = entryPrice + (atrValue * 1.0);
                target = entryPrice - (atrValue * 2.0);

                executeTrade(false, entryPrice, stopLoss, target, candles, i);
            }
        }

        printResults();
    }

    private void executeTrade(boolean isBuy,
                              double entry,
                              double stop,
                              double target,
                              List<Candle> candles,
                              int startIndex) {

        for (int j = startIndex + 1; j < candles.size(); j++) {

            double high = candles.get(j).getHigh();
            double low = candles.get(j).getLow();

            if (isBuy) {

                if (low <= stop) {
                    recordLoss(entry - stop);
                    return;
                }

                if (high >= target) {
                    recordWin(target - entry);
                    return;
                }

            } else {

                if (high >= stop) {
                    recordLoss(stop - entry);
                    return;
                }

                if (low <= target) {
                    recordWin(entry - target);
                    return;
                }
            }
        }
    }

    private void recordWin(double profit) {
        wins++;
        netProfit += profit;
        equity += profit;
        updateDrawdown();
    }

    private void recordLoss(double loss) {
        losses++;
        netProfit -= loss;
        equity -= loss;
        updateDrawdown();
    }

    private void updateDrawdown() {

        if (equity > peakEquity) {
            peakEquity = equity;
        }

        double drawdown = peakEquity - equity;

        if (drawdown > maxDrawdown) {
            maxDrawdown = drawdown;
        }
    }

    private void printResults() {

        int totalTrades = wins + losses;
        double winRate = totalTrades == 0 ? 0 :
                ((double) wins / totalTrades) * 100;

        System.out.println("\n===== BACKTEST RESULT =====");
        System.out.println("Total Trades: " + totalTrades);
        System.out.println("Wins: " + wins);
        System.out.println("Losses: " + losses);
        System.out.println("Win Rate: " + winRate + "%");
        System.out.println("Net Profit: " + netProfit);
        System.out.println("Max Drawdown: " + maxDrawdown);
    }
}