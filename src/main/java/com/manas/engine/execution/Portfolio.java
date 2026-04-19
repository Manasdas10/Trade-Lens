package com.manas.engine.execution;

import java.util.ArrayList;
import java.util.List;

public class Portfolio {

    private double capital = 100000;
    private List<Trade> trades = new ArrayList<>();

    public void recordTrade(Trade trade){

        trades.add(trade);

        capital += trade.getProfit();
    }

    public double getCapital(){
        return capital;
    }

    public int totalTrades(){
        return trades.size();
    }

    public List<Trade> getTrades(){
        return trades;
    }
}