package com.manas.engine.strategy;

import java.util.List;

import com.manas.engine.indicators.EMAIndicator;
import com.manas.engine.indicators.RSIIndicator;
import com.manas.engine.indicators.SMAIndicator;
import com.manas.engine.model.Candle;
import com.manas.engine.priceaction.ImpulseCandleDetector;
import com.manas.engine.smc.BreakOfStructureDetector;
import com.manas.engine.smc.MarketStructure;

public class EMACrossoverStrategy implements Strategy {

    private final EMAIndicator fastEma = new EMAIndicator(9);
    private final EMAIndicator slowEma = new EMAIndicator(15);
    private final EMAIndicator trendEma = new EMAIndicator(50);
    private final RSIIndicator rsi = new RSIIndicator(14);
    private final SMAIndicator sma = new SMAIndicator(20);

    private final MarketStructure structure = new MarketStructure();
    private final BreakOfStructureDetector bos = new BreakOfStructureDetector();
    private final ImpulseCandleDetector impulse = new ImpulseCandleDetector();

    @Override
    public Signal generateSignal(List<Candle> candles) {

        if (candles.size() < 30) {
            return Signal.HOLD;
        }

        double currFast = fastEma.calculate(candles);
        double currSlow = slowEma.calculate(candles);
        double trendValue = trendEma.calculate(candles);
        double currentRsi = rsi.calculate(candles);
        double smaValue = sma.calculate(candles);

        Candle lastCandle = candles.get(candles.size() - 1);
        double lastClose = lastCandle.getClose();

        MarketStructure.Trend trend = structure.detectTrend(candles);

        boolean bullishMomentum = currFast > currSlow;
        boolean bearishMomentum = currFast < currSlow;

        boolean bullishBOS = bos.bullishBOS(candles);
        boolean bearishBOS = bos.bearishBOS(candles);

        boolean bullishImpulse = impulse.isBullishImpulse(candles);
        boolean bearishImpulse = impulse.isBearishImpulse(candles);

        // ===== BUY =====
        if (trend == MarketStructure.Trend.BULLISH &&
            bullishMomentum &&
            bullishBOS &&
            bullishImpulse &&
            lastClose > trendValue &&
            lastClose > smaValue &&
            currentRsi > 50) {

            return Signal.BUY;
        }

        // ===== SELL =====
        if (trend == MarketStructure.Trend.BEARISH &&
            bearishMomentum &&
            bearishBOS &&
            bearishImpulse &&
            lastClose < trendValue &&
            lastClose < smaValue &&
            currentRsi < 45) {

            return Signal.SELL;
        }

        return Signal.HOLD;
    }
}