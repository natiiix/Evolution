using System;
using System.Collections.Generic;
using System.IO;

namespace Evolution
{
    public static class CSVParser
    {
        public static List<double[]> ParseFile(string filePath)
        {
            // Make sure file exists
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException();
            }

            // Create list to store the data
            List<double[]> dataMatrix = new List<double[]>();

            // Change decimal separator character to dot by changing the current culture to English (US)
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.GetCultureInfo("en-US");

            // Read from the file
            using (StreamReader sr = new StreamReader(filePath))
            {
                // Until the end of file is reached
                while (!sr.EndOfStream)
                {
                    // Read line and parse double values from it
                    double[] dValues = ParseLine(sr.ReadLine());

                    // If there were some values on the line
                    if (dValues.Length > 0)
                    {
                        // Append the array of double values to the list
                        dataMatrix.Add(dValues);
                    }
                }
            }

            // Return the list of values
            return dataMatrix;
        }

        public static double[] ParseLine(string str)
        {
            // Separate individual values on the line
            string[] strValues = str.Split(new char[] { ' ', ',', ';' }, StringSplitOptions.RemoveEmptyEntries);

            // Convert values from string to double
            double[] dValues = new double[strValues.Length];

            for (int i = 0; i < strValues.Length; i++)
            {
                if (!double.TryParse(strValues[i], out dValues[i]))
                {
                    // Double value parsing failed
                    throw new InvalidDataException();
                }
            }

            return dValues;
        }
    }
}