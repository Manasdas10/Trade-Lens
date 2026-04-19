package com.manas.engine.model;

public class Position {

    private final double entryPrice;
    private final double stopLoss;
    private final int quantity;

    public Position(double entryPrice, double stopLoss, int quantity) {
        this.entryPrice = entryPrice;
        this.stopLoss = stopLoss;
        this.quantity = quantity;
    }

    public double getEntryPrice() {
        return entryPrice;
    }

    public double getStopLoss() {
        return stopLoss;
    }

    public int getQuantity() {
        return quantity;
    }
}