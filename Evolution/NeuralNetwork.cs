//using System;

//namespace Evolution
//{
//    public class NeuralNetwork
//    {
//        private static Random rand = new Random();

//        public double AverageDeviation;
//        public int IterationsAlive;

//        private int m_inputCount;
//        private int m_hiddenCount;
//        private int m_outputCount;

//        // [layer][node][connection]
//        private double[][][] m_weights;

//        private const double WEIGHT_MULTIPLIER = 100.0;

//        public NeuralNetwork(int inputCount, int hiddenCount, int outputCount)
//        {
//            AverageDeviation = 0.0;
//            IterationsAlive = 0;

//            m_inputCount = inputCount;
//            m_hiddenCount = hiddenCount;
//            m_outputCount = outputCount;

//            m_weights = new double[2][][] { new double[hiddenCount][], new double[outputCount][] };

//            // Set up weights between input layer and hidden layer
//            for (int i = 0; i < hiddenCount; i++)
//            {
//                // Hidden nodes are connected to input nodes and bias node
//                m_weights[0][i] = new double[inputCount + 1];
//            }

//            // Set up weights between hidden layer and output layer
//            for (int i = 0; i < outputCount; i++)
//            {
//                // Output nodes are connected to hidden nodes and bias node
//                m_weights[1][i] = new double[hiddenCount + 1];
//            }

//            RandomizeWeights();
//        }

//        private NeuralNetwork(int inputCount, int hiddenCount, int outputCount, double[][][] weights)
//        {
//            AverageDeviation = 0.0;
//            IterationsAlive = 0;

//            m_inputCount = inputCount;
//            m_hiddenCount = hiddenCount;
//            m_outputCount = outputCount;

//            m_weights = weights;
//        }

//        private void RandomizeWeights()
//        {
//            // For each layer
//            for (int i = 0; i < m_weights.GetLength(0); i++)
//                // For each node within the layer
//                for (int j = 0; j < m_weights[i].GetLength(0); j++)
//                    // For each connection to the node
//                    for (int k = 0; k < m_weights[i][j].GetLength(0); k++)
//                        // Set each weight to a random value between -1.0 and 1.0 (multiplied by 100.0 to fit the hyperbolic tangent function better)
//                        m_weights[i][j][k] = ((rand.NextDouble() * 2.0) - 1.0) * WEIGHT_MULTIPLIER;
//        }

//        public double[] ProcessData(double[] input)
//        {
//            // Make sure the number of input values matches the NN structure
//            if (input.Length != m_inputCount)
//            {
//                throw new ArgumentException("\nUnexpected input array length! (Length = " + input.Length.ToString() + ")\nExpected " + m_inputCount.ToString() + " input values!");
//            }

//            // Compute values of hidden nodes
//            double[] hiddenValues = new double[m_hiddenCount];

//            for (int i = 0; i < m_hiddenCount; i++)
//            {
//                hiddenValues[i] = 0.0;

//                for (int j = 0; j < m_inputCount + 1; j++)
//                {
//                    // Apply bias
//                    if (j == m_inputCount)
//                    {
//                        hiddenValues[i] += m_weights[0][i][j];
//                    }
//                    else
//                    {
//                        hiddenValues[i] += input[j] * m_weights[0][i][j];
//                    }
//                }

//                // Apply hyperbolic tangent
//                hiddenValues[i] = Math.Tanh(hiddenValues[i]);
//            }

//            // Compute values of output nodes
//            double[] outputValues = new double[m_outputCount];

//            for (int i = 0; i < m_outputCount; i++)
//            {
//                outputValues[i] = 0.0;

//                for (int j = 0; j < m_hiddenCount + 1; j++)
//                {
//                    // Apply bias
//                    if (j == m_hiddenCount)
//                    {
//                        outputValues[i] += m_weights[1][i][j];
//                    }
//                    else
//                    {
//                        outputValues[i] += hiddenValues[j] * m_weights[1][i][j];
//                    }
//                }

//                // Apply hyperbolic tangent
//                outputValues[i] = Math.Tanh(outputValues[i]);
//            }

//            // Return computed output values
//            return outputValues;
//        }

//        public void StoreDeviation(double deviation)
//        {
//            // Update the average deviation and increment the iteration counter
//            AverageDeviation = ((AverageDeviation * IterationsAlive) + deviation) / ++IterationsAlive;
//        }

//        public NeuralNetwork ShakeWeights()
//        {
//            // Create a deep clone of the weights array
//            double[][][] shakenWeights = new double[m_weights.GetLength(0)][][];

//            for (int i = 0; i < m_weights.GetLength(0); i++)
//            {
//                shakenWeights[i] = new double[m_weights[i].GetLength(0)][];

//                for (int j = 0; j < m_weights[i].GetLength(0); j++)
//                {
//                    shakenWeights[i][j] = new double[m_weights[i][j].GetLength(0)];

//                    for (int k = 0; k < m_weights[i][j].GetLength(0); k++)
//                    {
//                        shakenWeights[i][j][k] = m_weights[i][j][k];
//                    }
//                }
//            }

//            // For each layer
//            for (int i = 0; i < shakenWeights.GetLength(0); i++)
//                // For each node within the layer
//                for (int j = 0; j < shakenWeights[i].GetLength(0); j++)
//                    // For each connection to the node
//                    for (int k = 0; k < shakenWeights[i][j].GetLength(0); k++)
//                        // Give each weight a random "shake"
//                        shakenWeights[i][j][k] += ((rand.NextDouble() * AverageDeviation * 2.0) - AverageDeviation) * WEIGHT_MULTIPLIER;

//            // Return a new NN with shaken weights
//            return new NeuralNetwork(m_inputCount, m_hiddenCount, m_outputCount, shakenWeights);
//        }
//    }
//}