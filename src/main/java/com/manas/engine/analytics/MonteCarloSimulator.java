package com.manas.engine.analytics;

import java.util.List;
import java.util.Random;

import com.manas.engine.execution.Trade;

public class MonteCarloSimulator {

    public static double simulate(List<Trade> trades) {

        if (trades.isEmpty()) return 0;

        Random random = new Random();

        double equity = 100000;

        for (int i = 0; i < trades.size(); i++) {

            Trade trade = trades.get(random.nextInt(trades.size()));

            equity += trade.getProfit();
        }

        return equity;
    }
}