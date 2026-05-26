package com.manas.engine.ai;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class AIPredictor {

    public static String runPrediction(double closePrice) {

        try {

            ProcessBuilder pb =
                    new ProcessBuilder(
                            "python",
                            "ai-python/predictor.py",
                            String.valueOf(closePrice)
                    );

            Process process = pb.start();

            BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(
                                    process.getInputStream()
                            )
                    );

            String prediction = reader.readLine();

            process.waitFor();

            return prediction;

        } catch (Exception e) {

            System.out.println("AI ERROR: "
                    + e.getMessage());

            e.printStackTrace();
        }

        return "HOLD";
    }
}