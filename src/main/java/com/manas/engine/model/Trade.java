package com.manas.engine.model;

public class Trade {

    private final double entryPrice;
    private double exitPrice;
    private final boolean isLong;
    private boolean closed;
    private double profit;

    public Trade(double entryPrice, boolean isLong) {
        this.entryPrice = entryPrice;
        this.isLong = isLong;
        this.closed = false;
    }

    public void close(double exitPrice) {
        this.exitPrice = exitPrice;
        this.closed = true;

        if (isLong) {
            profit = exitPrice - entryPrice;
        } else {
            profit = entryPrice - exitPrice;
        }
    }

    public double getEntryPrice() {
        return entryPrice;
    }

    public double getExitPrice() {
        return exitPrice;
    }

    public double getProfit() {
        return profit;
    }

    public boolean isClosed() {
        return closed;
    }

    public boolean isLong() {
        return isLong;
    }
}