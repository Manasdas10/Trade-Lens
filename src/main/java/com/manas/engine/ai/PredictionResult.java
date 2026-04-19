package com.manas.engine.ai;

public class PredictionResult {

    private double predictedPrice;
    private double confidence;

    public PredictionResult(double predictedPrice, double confidence){
        this.predictedPrice = predictedPrice;
        this.confidence = confidence;
    }

    public double getPredictedPrice(){
        return predictedPrice;
    }

    public double getConfidence(){
        return confidence;
    }
}