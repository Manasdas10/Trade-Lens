package com.manas.engine.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import com.manas.engine.model.Candle;

public class CSVLoader {

    public static List<Candle> load(String filePath) {

        List<Candle> candles = new ArrayList<>();

        try {

            BufferedReader br = new BufferedReader(new FileReader(filePath));

            String line;

            // Skip header
            br.readLine();

            while ((line = br.readLine()) != null) {

                String[] data = line.split(",");

                String time = data[0];

                double open = Double.parseDouble(data[1]);
                double high = Double.parseDouble(data[2]);
                double low = Double.parseDouble(data[3]);
                double close = Double.parseDouble(data[4]);

                long volume = (long) Double.parseDouble(data[5]);

                Candle candle = new Candle(
                        time,
                        open,
                        high,
                        low,
                        close,
                        volume
                );

                candles.add(candle);
            }

            br.close();

        } catch (Exception e) {

            System.out.println("CSV LOAD ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        return candles;
    }
}