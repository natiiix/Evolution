using System;
using System.Linq;

namespace Evolution
{
    public class DeepNeuralNetwork
    {
        private static Random rand = new Random();

        public double AverageDeviation;
        public int IterationsAlive;

        private readonly int[] LayerNodes;

        private int INPUT_NODES { get => LayerNodes.First(); }
        private int OUTPUT_NODES { get => LayerNodes.Last(); }

        // [layer][node][connection]
        private double[][][] Weights;

        private const double WEIGHT_MULTIPLIER = 1.0;

        public DeepNeuralNetwork(int inputCount, int outputCount, params int[] hiddenCounts)
        {
            AverageDeviation = 0.0;
            IterationsAlive = 0;

            // Number of nodes in each layer
            // Bias nodes are not counted in
            LayerNodes = new int[hiddenCounts.Length + 2];

            LayerNodes[0] = inputCount;

            for (int i = 0; i < hiddenCounts.Length; i++)
            {
                LayerNodes[i + 1] = hiddenCounts[i];
            }

            LayerNodes[LayerNodes.Length - 1] = outputCount;

            // Initialize the weights structure
            // Every layer is connected to the previous layer except for the input layer
            Weights = new double[LayerNodes.Length - 1][][];

            for (int i = 0; i < Weights.Length; i++)
            {
                // Number of nodes in the previous layer plus the bias node
                int prevLayerNodes = LayerNodes[i] + 1;

                // Nubmer of nodes in this layer, not counting in the bias node
                int thisLayerNodes = LayerNodes[i + 1];

                // Node level
                Weights[i] = new double[thisLayerNodes][];

                // Individual connection level
                for (int j = 0; j < thisLayerNodes; j++)
                {
                    Weights[i][j] = new double[prevLayerNodes];
                }
            }

            RandomizeWeights();
        }

        private DeepNeuralNetwork(int[] layerNodes, double[][][] weights)
        {
            AverageDeviation = 0.0;
            IterationsAlive = 0;

            LayerNodes = layerNodes;
            Weights = weights;
        }

        private void RandomizeWeights()
        {
            // For each layer
            for (int i = 0; i < Weights.GetLength(0); i++)
                // For each node within the layer
                for (int j = 0; j < Weights[i].GetLength(0); j++)
                    // For each connection to the node
                    for (int k = 0; k < Weights[i][j].GetLength(0); k++)
                        // Set each weight to a random value between -1.0 and 1.0 (multiplied by 100.0 to fit the hyperbolic tangent function better)
                        Weights[i][j][k] = ((rand.NextDouble() * 2.0) - 1.0) * WEIGHT_MULTIPLIER;
        }

        public double[] ProcessData(double[] input)
        {
            // Make sure the number of input values matches the NN structure
            if (input.Length != INPUT_NODES)
            {
                throw new ArgumentException("\nUnexpected input array length! (Length = " + input.Length.ToString() + ")\nExpected " + INPUT_NODES.ToString() + " input values!");
            }

            // The first hidden layer uses the input layer as source
            double[] prevLayer = input;

            // Iterate through the layers, skip the input layer
            for (int i = 1; i < LayerNodes.Length; i++)
            {
                double[] thisLayer = new double[LayerNodes[i]];

                // Calculate the value of each node
                for (int j = 0; j < thisLayer.Length; j++)
                {
                    double nodeValue = 0;

                    // Iterate through the nodes of the previous layer
                    for (int k = 0; k < prevLayer.Length; k++)
                    {
                        nodeValue += prevLayer[k] * Weights[i - 1][j][k];
                    }

                    // Add the weighted bias value
                    nodeValue += Weights[i - 1][j].Last();

                    // Apply the hyperbolic tangent function and copy the node value
                    thisLayer[j] = Math.Tanh(nodeValue);
                }

                // This layer will be used as a source for the next layer
                prevLayer = thisLayer;
            }

            // Return computed output values
            return prevLayer;
        }

        public void StoreDeviation(double deviation)
        {
            // Update the average deviation and increment the iteration counter
            AverageDeviation = ((AverageDeviation * IterationsAlive) + deviation) / ++IterationsAlive;
        }

        public DeepNeuralNetwork ShakeWeights()
        {
            // Create a deep clone of the weights array
            double[][][] shakenWeights = new double[Weights.GetLength(0)][][];

            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                shakenWeights[i] = new double[Weights[i].GetLength(0)][];

                for (int j = 0; j < Weights[i].GetLength(0); j++)
                {
                    shakenWeights[i][j] = new double[Weights[i][j].GetLength(0)];

                    for (int k = 0; k < Weights[i][j].GetLength(0); k++)
                    {
                        shakenWeights[i][j][k] = Weights[i][j][k];
                    }
                }
            }

            // For each layer
            for (int i = 0; i < shakenWeights.GetLength(0); i++)
                // For each node within the layer
                for (int j = 0; j < shakenWeights[i].GetLength(0); j++)
                    // For each connection to the node
                    for (int k = 0; k < shakenWeights[i][j].GetLength(0); k++)
                        // Give each weight a random "shake"
                        shakenWeights[i][j][k] += ((rand.NextDouble() * AverageDeviation * 2.0) - AverageDeviation) * WEIGHT_MULTIPLIER / OUTPUT_NODES;

            // Return a new NN with shaken weights
            return new DeepNeuralNetwork(LayerNodes, shakenWeights);
        }
    }
}