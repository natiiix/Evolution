using System;
using System.Linq;
using System.IO;

namespace Evolution
{
    public class Program
    {
        private static Random rand;
        private static DateTime dtLastIteration;

        private static double[][] sourceData;

        private const int INPUT_SIZE = 4;
        private const int HIDDEN_SIZE = 6;
        private const int OUTPUT_SIZE = 3;
        private const int COMPLEXITY = INPUT_SIZE * HIDDEN_SIZE * OUTPUT_SIZE;
        private const int NETWORK_SIZE = 16 * COMPLEXITY;
        private const int NETWORK_QUARTER = NETWORK_SIZE / 4;
        private const int ITERATION_LIMIT = 64 * COMPLEXITY;
        private const int SUB_ITERATION_COUNT = 256;

        private static void Main(string[] args)
        {
            if (NETWORK_SIZE % 4 > 0)
            {
                throw new Exception("Network size must be a multiple of 4!");
            }

            if (!ReadFileDataToMatrix("data.csv", out sourceData))
            {
                throw new FileNotFoundException();
            }

            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.GetCultureInfo("en-US");
            rand = new Random();

            NeuralNetwork[] nn = new NeuralNetwork[NETWORK_SIZE];

            // Construct initial neural networks
            for (int i = 0; i < nn.Length; i++)
            {
                nn[i] = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            }

            dtLastIteration = DateTime.Now;
            // Perform training iterations
            for (int i = 0; i < ITERATION_LIMIT; i++)
            {
                PerformEvolution(ref nn, i);
            }

            // Print training summary
            Console.WriteLine(Environment.NewLine + "The top quarter neural networks have " + GetAverageDeviation(nn).ToString() + " average deviation.");
            Console.WriteLine("The best neural network has " + nn[0].AverageDeviation.ToString() + " deviation over " + (nn[0].IterationsAlive / SUB_ITERATION_COUNT).ToString() + " iterations.");

            // Let the user try the NN with his own input values
            while (true)
            {
                Console.Write(Environment.NewLine + "Enter input values: ");
                string strInput = Console.ReadLine();

                // Empty input is an exit request
                if (strInput == string.Empty)
                {
                    break;
                }

                // Separate individual values on the line
                string[] strValues = strInput.Split(new char[] { ' ', ',', ';' }, StringSplitOptions.RemoveEmptyEntries);

                if (strValues.Length != INPUT_SIZE)
                {
                    throw new ArgumentException("Unexpected number of input values!");
                }

                // Convert values from string to double
                double[] dValues = new double[strValues.Length];

                for (int i = 0; i < strValues.Length; i++)
                {
                    if (!double.TryParse(strValues[i], out dValues[i]))
                    {
                        // Double value parsing failed
                        throw new ArgumentException("Invalid input value!");
                    }
                }

                // Get output values from the NN
                double[] output = nn[0].ProcessData(dValues);

                Console.Write("Output: ");

                for (int i = 0; i < output.Length; i++)
                {
                    Console.Write(output[i].ToString() + " ");
                }

                Console.WriteLine();
            }
        }

        // Keep 1/4 of the NNs as survivors
        // Replace 2/4 of the NNs with modifications of the survivors
        // Fill the remaining 1/4 with completely new NNs
        private static void PerformEvolution(ref NeuralNetwork[] networks, int iterationIdx)
        {
            for (int i = 0; i < SUB_ITERATION_COUNT; i++)
            {
                FeedInputToNetworks(ref networks, rand.Next(sourceData.GetLength(0)));
            }

            // Order NNs by their average deviation
            networks = networks.OrderBy(x => x.AverageDeviation).ToArray();

            if ((DateTime.Now - dtLastIteration) >= new TimeSpan(0, 0, 1))
            {
                dtLastIteration = DateTime.Now;
                // Print average deviation for this iteration
                Console.WriteLine("Iteration: " + (iterationIdx + 1).ToString().PadLeft(ITERATION_LIMIT.ToString().Length) + " - Deviation: " + GetAverageDeviation(networks).ToString());
            }

            // Don't modify the NNs on the last iteration
            if (iterationIdx == ITERATION_LIMIT - 1)
            {
                return;
            }

            // Replace 2/4 of NNs with children of survivors
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < NETWORK_QUARTER; j++)
                {
                    // NN to be replaced with one of the children
                    ref NeuralNetwork nn = ref networks[((i + 1) * NETWORK_QUARTER) + j];

                    // Don't use networks with no deviation as parents
                    // They would just produce their exact clones and that's pointless
                    if (networks[j].AverageDeviation == 0.0)
                    {
                        // Generate a completely new NN with random weights instead
                        nn = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
                    }
                    else
                    {
                        // Replace the NN with a slightly modifier version of one of the survivors
                        nn = networks[j].ShakeWeights();
                    }
                }
            }

            // Replace the last 1/4 with new NNs with randomly generated weights
            for (int j = 0; j < NETWORK_QUARTER; j++)
            {
                // NN to be replaced with one of the children
                networks[(3 * NETWORK_QUARTER) + j] = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            }
        }

        private static void FeedInputToNetworks(ref NeuralNetwork[] networks, int sourceIdx)
        {
            double[] inputValues = GetInputArray(sourceIdx);

            // Process input data by each of the NNs
            // Calculate a deviation based on the deviation between predicted output and expected output
            // Then update the average deviation for each NN
            for (int j = 0; j < networks.Length; j++)
            {
                networks[j].StoreDeviation(GetDeviation(sourceIdx, networks[j].ProcessData(inputValues)));
            }
        }

        // Calculates the average deviation of the top 1/4 of the NNs
        private static double GetAverageDeviation(NeuralNetwork[] networks)
        {
            double deviationSum = 0.0;

            // Sum average deviations form all the NNs in the top quarter
            for (int j = 0; j < NETWORK_QUARTER; j++)
            {
                deviationSum += networks[j].AverageDeviation;
            }

            return deviationSum / NETWORK_QUARTER;
        }

        /*// Generates random input values
        private static double[] GenerateInput()
        {
            double[] input = new double[INPUT_SIZE];

            for (int i = 0; i < input.Length; i++)
            {
                //input[i] = (rand.NextDouble() * 2.0) - 1.0;
                //input[i] = (rand.NextDouble() > 0.5 ? 1.0 : -1.0);
            }

            return input;
        }

        // Computes the deviation between the predicted output and the correct output
        private static double GetDeviation(double[] input, double[] output)
        {
            // Handle wrong size of input array
            if (input.Length != INPUT_SIZE)
            {
                throw new ArgumentException("Unexpected input array size!");
            }

            // Handle wrong size of output array
            if (output.Length != OUTPUT_SIZE)
            {
                throw new ArgumentException("Unexpected output array size!");
            }

            double[] correctOutput = new double[OUTPUT_SIZE];
            correctOutput[0] = (input[0] != input[1] ? 1.0 : -1.0);

            // Calculate the deviation of each output node and sum them
            double deviation = 0.0;

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                deviation += Math.Abs(correctOutput[i] - output[i]);
            }

            return deviation;
        }*/

        // Copies the input part of the source data into a separate array
        private static double[] GetInputArray(int sourceIdx)
        {
            double[] input = new double[INPUT_SIZE];

            for (int i = 0; i < INPUT_SIZE; i++)
            {
                input[i] = sourceData[sourceIdx][i];
            }

            return input;
        }

        private static double GetDeviation(int sourceIdx, double[] predictedOutput)
        {
            // Handle wrong size of output array
            if (predictedOutput.Length != OUTPUT_SIZE)
            {
                throw new ArgumentException("Unexpected output array size!");
            }

            // Calculate the deviation of each output node and sum them
            double deviation = 0.0;

            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                deviation += Math.Abs(sourceData[sourceIdx][INPUT_SIZE + i] - predictedOutput[i]);
            }

            return deviation;
        }

        private static bool ReadFileDataToMatrix(string filePath, out double[][] dataMatrix)
        {
            // Make sure file exists
            if (!File.Exists(filePath))
            {
                dataMatrix = new double[0][];
                return false;
            }

            // Change decimal separator character to dot by changing the current culture to English (US)
            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.GetCultureInfo("en-US");

            // Create list to store the data
            System.Collections.Generic.List<double[]> dataList = new System.Collections.Generic.List<double[]>();

            // Read from the file
            using (StreamReader sr = new StreamReader(filePath))
            {
                // Until the end of file is reached
                while (!sr.EndOfStream)
                {
                    // Read line and parse double values from it
                    double[] dValues = ParseCSV(sr.ReadLine());

                    // If there were some values on the line
                    if (dValues.Length > 0)
                    {
                        // Push the array of double values to the list
                        dataList.Add(dValues);
                    }
                }
            }

            dataMatrix = dataList.ToArray();
            return true;
        }

        private static double[] ParseCSV(string str)
        {
            // Separate individual values on the line
            string[] strValues = str.Split(new char[] { ' ', ',', ';' }, StringSplitOptions.RemoveEmptyEntries);

            // Only work with non-emtpy lines
            if (strValues.Length > 0)
            {
                // Convert values from string to double
                double[] dValues = new double[strValues.Length];

                for (int i = 0; i < strValues.Length; i++)
                {
                    if (!double.TryParse(strValues[i], out dValues[i]))
                    {
                        // Double value parsing failed
                        return new double[0];
                    }
                }

                return dValues;
            }

            // Empty line
            return new double[0];
        }
    }
}