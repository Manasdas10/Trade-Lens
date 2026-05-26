package com.manas.engine.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

import com.manas.engine.model.Candle;

import yahoofinance.Stock;
import yahoofinance.YahooFinance;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

public class YahooDataFetcher {

    public static List<Candle> fetch(
            String symbol,
            Interval interval) throws IOException {

        Calendar from = Calendar.getInstance();
        Calendar to = Calendar.getInstance();

        from.add(Calendar.MONTH, -6);

        Stock stock = YahooFinance.get(symbol);

        List<HistoricalQuote> history =
                stock.getHistory(from, to, interval);

        List<Candle> candles = new ArrayList<>();

        for (HistoricalQuote q : history) {

            if (q.getOpen() == null ||
                q.getHigh() == null ||
                q.getLow() == null ||
                q.getClose() == null ||
                q.getVolume() == null ||
                q.getDate() == null) {

                continue;
            }

            String time = q.getDate().getTime().toString();

            Candle candle = new Candle(
                    time,
                    q.getOpen().doubleValue(),
                    q.getHigh().doubleValue(),
                    q.getLow().doubleValue(),
                    q.getClose().doubleValue(),
                    q.getVolume()
            );

            candles.add(candle);
        }

        return candles;
    }
}