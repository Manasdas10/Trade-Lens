package com.manas.engine.strategy;

import java.util.List;

import com.manas.engine.indicators.EMAIndicator;
import com.manas.engine.indicators.RSIIndicator;
import com.manas.engine.model.Candle;
import com.manas.engine.priceaction.ImpulseCandleDetector;
import com.manas.engine.smc.BreakOfStructureDetector;
import com.manas.engine.smc.MarketStructure;

public class EMACrossoverStrategy implements Strategy {

    private final EMAIndicator fastEma = new EMAIndicator(9);
    private final EMAIndicator slowEma = new EMAIndicator(15);
    private final EMAIndicator trendEma = new EMAIndicator(50);
    private final RSIIndicator rsi = new RSIIndicator(14);

    private final MarketStructure structure = new MarketStructure();
    private final BreakOfStructureDetector bos = new BreakOfStructureDetector();
    private final ImpulseCandleDetector impulse = new ImpulseCandleDetector();

    @Override
    public Signal generateSignal(List<Candle> candles) {

        if (candles.size() < 120) {
            return Signal.HOLD;
        }

        double currFast = fastEma.calculate(candles);
        double currSlow = slowEma.calculate(candles);
        double trendValue = trendEma.calculate(candles);
        double currentRsi = rsi.calculate(candles);

        Candle lastCandle = candles.get(candles.size() - 1);
        double lastClose = lastCandle.getClose();

        MarketStructure.Trend trend = structure.detectTrend(candles);

        boolean bullishMomentum = currFast > currSlow;
        boolean bearishMomentum = currFast < currSlow;

        boolean bullishBOS = bos.bullishBOS(candles);
        boolean bearishBOS = bos.bearishBOS(candles);

        boolean bullishImpulse = impulse.isBullishImpulse(candles);
        boolean bearishImpulse = impulse.isBearishImpulse(candles);

        // ===== BUY SETUP =====
        if (trend == MarketStructure.Trend.BULLISH &&
            bullishMomentum &&
            bullishBOS &&
            bullishImpulse &&
            lastClose > trendValue &&
            currentRsi > 50) {

            return Signal.BUY;
        }

        // ===== SELL SETUP =====
        if (trend == MarketStructure.Trend.BEARISH &&
            bearishMomentum &&
            bearishBOS &&
            bearishImpulse &&
            lastClose < trendValue &&
            currentRsi < 50) {

            return Signal.SELL;
        }

        return Signal.HOLD;
    }
}