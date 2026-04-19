package com.manas.engine.strategy;

import java.util.List;

import com.manas.engine.indicators.EMAIndicator;
import com.manas.engine.indicators.RSIIndicator;
import com.manas.engine.indicators.VWAPIndicator;
import com.manas.engine.indicators.VolumeSpikeIndicator;
import com.manas.engine.model.Candle;

public class ScalpingStrategy implements Strategy {

    private final EMAIndicator fastEma;
    private final EMAIndicator slowEma;
    private final RSIIndicator rsi;
    private final VWAPIndicator vwap;
    private final VolumeSpikeIndicator volumeSpike;

    public ScalpingStrategy() {
        this.fastEma = new EMAIndicator(9);
        this.slowEma = new EMAIndicator(15);
        this.rsi = new RSIIndicator(14);
        this.vwap = new VWAPIndicator();
        this.volumeSpike = new VolumeSpikeIndicator();
    }

    @Override
    public Signal generateSignal(List<Candle> candles) {

        if (candles == null || candles.size() < 30) {
            return Signal.HOLD;
        }

        double currFast = fastEma.calculate(candles);
        double currSlow = slowEma.calculate(candles);

        double prevFast = fastEma.calculate(
                candles.subList(0, candles.size() - 1)
        );

        double prevSlow = slowEma.calculate(
                candles.subList(0, candles.size() - 1)
        );

        double currentRsi = rsi.calculate(candles);
        double currentVwap = vwap.calculate(candles);

        Candle lastCandle = candles.get(candles.size() - 1);
        boolean volumeStrong = volumeSpike.isVolumeSpike(candles);

        // BUY Logic
        if (prevFast <= prevSlow &&
                currFast > currSlow &&
                currentRsi > 55 &&
                lastCandle.getClose() > currentVwap &&
                volumeStrong) {

            return Signal.BUY;
        }

        // SELL Logic
        if (prevFast >= prevSlow &&
                currFast < currSlow &&
                currentRsi < 45 &&
                lastCandle.getClose() < currentVwap &&
                volumeStrong) {

            return Signal.SELL;
        }

        return Signal.HOLD;
    }
}