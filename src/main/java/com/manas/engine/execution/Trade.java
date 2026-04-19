package com.manas.engine.execution;

public class Trade {

    private double entryPrice;
    private double exitPrice;
    private boolean win;

    public Trade(double entryPrice, double exitPrice, boolean win){
        this.entryPrice = entryPrice;
        this.exitPrice = exitPrice;
        this.win = win;
    }

    public double getProfit(){
        return exitPrice - entryPrice;
    }

    public boolean isWin(){
        return win;
    }
}