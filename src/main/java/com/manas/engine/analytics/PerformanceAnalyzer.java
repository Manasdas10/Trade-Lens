package com.manas.engine.analytics;

import java.util.List;

import com.manas.engine.execution.Trade;

public class PerformanceAnalyzer {

    public static double calculateSharpe(List<Trade> trades){

        if(trades.isEmpty()) return 0;

        double avg = trades.stream()
                .mapToDouble(Trade::getProfit)
                .average()
                .orElse(0);

        double variance = trades.stream()
                .mapToDouble(t -> Math.pow(t.getProfit() - avg,2))
                .average()
                .orElse(0);

        double std = Math.sqrt(variance);

        if(std == 0) return 0;

        return avg/std;
    }
}