package com.manas.engine.indicators;

import java.util.List;

import com.manas.engine.model.Candle;

public class VolumeSpikeIndicator {

    public boolean isVolumeSpike(List<Candle> candles) {

        if (candles.size() < 20) return false;

        double sum = 0;
        int size = candles.size();

        for (int i = size - 20; i < size - 1; i++) {
            sum += candles.get(i).getVolume();
        }

        double avg = sum / 20;
        double currentVolume = candles.get(size - 1).getVolume();

        return currentVolume > avg * 1.5;
    }
}