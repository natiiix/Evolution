using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Evolution
{
    public class Program
    {
        private static Random rand;
        private static DateTime dtLastIteration;

        private readonly static List<double[]> sourceData = CSVParser.ParseFile("data.csv");

        private readonly static int[] inputColumns = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        private readonly static int[] outputColumns = new int[] { 9, 10 };

        // Number of nodes in each of the hidden layers
        private readonly static int[] hiddenLayersNodes = new int[] { 12, 8 };

        //private static readonly int COMPLEXITY = inputColumns.Length * hiddenLayersNodes.Aggregate((x, y) => x * y) * outputColumns.Length;

        private static readonly int NETWORK_QUARTER = 32;
        private static readonly int NETWORK_SIZE = NETWORK_QUARTER * 4;
        private const int ITERATION_LIMIT = 4096;
        private const int SUB_ITERATION_COUNT = 64;

        private static void Main(string[] args)
        {
            if (NETWORK_SIZE % 4 > 0)
            {
                throw new Exception("Network size must be a multiple of 4!");
            }

            System.Threading.Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.GetCultureInfo("en-US");
            rand = new Random();

            DeepNeuralNetwork[] nn = new DeepNeuralNetwork[NETWORK_SIZE];

            // Construct initial neural networks
            for (int i = 0; i < nn.Length; i++)
            {
                nn[i] = CreateNewNN();
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
                double[] userInput = CSVParser.ParseLine(strInput);

                // Get output values from the NN
                double[] predictedOutput = nn[0].ProcessData(userInput);

                Console.WriteLine("Output: " + string.Join(" ", predictedOutput.AsEnumerable().Select(x => x.ToString())));
            }
        }

        private static DeepNeuralNetwork CreateNewNN() => new DeepNeuralNetwork(inputColumns.Length, outputColumns.Length, hiddenLayersNodes);

        public static double[] ExtractColumns(double[] data, int[] columns)
        {
            double[] extractedData = new double[columns.Length];

            for (int i = 0; i < columns.Length; i++)
            {
                extractedData[i] = data[columns[i]];
            }

            return extractedData;
        }

        // Keep 1/4 of the NNs as survivors
        // Replace 2/4 of the NNs with modifications of the survivors
        // Fill the remaining 1/4 with completely new NNs
        private static void PerformEvolution(ref DeepNeuralNetwork[] networks, int iterationIdx)
        {
            for (int i = 0; i < SUB_ITERATION_COUNT; i++)
            {
                FeedInputToNetworks(networks, rand.Next(sourceData.Count));
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
                    ref DeepNeuralNetwork nn = ref networks[((i + 1) * NETWORK_QUARTER) + j];

                    // Don't use networks with no deviation as parents
                    // They would just produce their exact clones and that's pointless
                    if (networks[j].AverageDeviation == 0.0)
                    {
                        // Generate a completely new NN with random weights instead
                        nn = CreateNewNN();
                    }
                    else
                    {
                        // Replace the NN with a slightly modifier version of one of the survivors
                        nn = networks[j].ShakeWeights();
                    }
                }
            }

            // Replace the last 1/4 with new NNs with randomly generated weights
            for (int i = 0; i < NETWORK_QUARTER; i++)
            {
                // NN to be replaced with one of the children
                networks[(3 * NETWORK_QUARTER) + i] = CreateNewNN();
            }
        }

        private static void FeedInputToNetworks(DeepNeuralNetwork[] networks, int sourceIdx)
        {
            double[] sourceRow = sourceData[sourceIdx];

            double[] inputValues = ExtractColumns(sourceRow, inputColumns);
            double[] outputValues = ExtractColumns(sourceRow, outputColumns);

            // Process input data by each of the NNs
            // Calculate a deviation based on the deviation between predicted output and expected output
            // Then update the average deviation for each NN
            Parallel.ForEach(networks, x => x.StoreDeviation(GetDeviation(x.ProcessData(inputValues), outputValues)));
        }

        // Calculates the average deviation of the top 1/4 of the NNs
        private static double GetAverageDeviation(DeepNeuralNetwork[] networks)
        {
            double deviationSum = 0.0;

            // Sum average deviations form all the NNs in the top quarter
            for (int i = 0; i < NETWORK_QUARTER; i++)
            {
                deviationSum += networks[i].AverageDeviation;
            }

            return deviationSum / NETWORK_QUARTER;
        }

        /*// Generates random input values
        private static double[] GenerateInput()
        {
            double[] input = new double[inputColumns.Length];

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
            if (input.Length != inputColumns.Length)
            {
                throw new ArgumentException("Unexpected input array size!");
            }

            // Handle wrong size of output array
            if (output.Length != outputColumns.Length)
            {
                throw new ArgumentException("Unexpected output array size!");
            }

            double[] correctOutput = new double[outputColumns.Length];
            correctOutput[0] = (input[0] != input[1] ? 1.0 : -1.0);

            // Calculate the deviation of each output node and sum them
            double deviation = 0.0;

            for (int i = 0; i < outputColumns.Length; i++)
            {
                deviation += Math.Abs(correctOutput[i] - output[i]);
            }

            return deviation;
        }*/

        // Copies the input part of the source data into a separate array
        //private static double[] GetInputArray(int sourceIdx)
        //{
        //    double[] input = new double[inputColumns.Length];

        //    for (int i = 0; i < inputColumns.Length; i++)
        //    {
        //        input[i] = sourceData[sourceIdx][i];
        //    }

        //    return input;
        //}

        private static double GetDeviation(double[] predictedOutput, double[] expectedOutput)
        {
            if (predictedOutput.Length != expectedOutput.Length)
            {
                throw new ArgumentException();
            }

            // Calculate the deviation of each output node and sum them
            double deviation = 0.0;

            for (int i = 0; i < predictedOutput.Length; i++)
            {
                deviation += Math.Abs(predictedOutput[i] - expectedOutput[i]);
            }

            return deviation;
        }
    }
}