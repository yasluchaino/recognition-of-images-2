using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;


namespace AIMLTGBot
{
    public class StudentNetwork : BaseNetwork
    {
        private class NNLayer
        {
            public double[,] weights { get; set; }
            public double[] biases { get; set; }
        }

        public Stopwatch stopWatch = new Stopwatch();

        private List<NNLayer> layers;

        private int batchSize;

        private double learningRate;

        public StudentNetwork(int[] structure, int batchSize = 10, double learningRate = 0.1)
        {
            Random random = new Random();
            layers = new List<NNLayer>();

            this.batchSize = batchSize;
            this.learningRate = learningRate;

            for (int i = 0; i < structure.Length - 1; i++)
            {
                var newLayer = new NNLayer
                {
                    weights = new double[structure[i], structure[i + 1]],
                    biases = new double[structure[i + 1]]
                };

                for (int j = 0; j < newLayer.biases.Length; j++)
                {
                    newLayer.biases[j] = random.NextDouble() * 0.2 - 0.1;

                    for (int k = 0; k < newLayer.weights.GetLength(0); k++)
                        newLayer.weights[k, j] = random.NextDouble() * 0.2 - 0.1;
                }

                layers.Add(newLayer);
            }
        }

        private double Activation(double x) => 1 / (1 + Math.Exp(-x));

        private double ActivationDerivative(double y) => y * (1 - y);

        public override int Train(Sample sample, double acceptableError, bool parallel, double learningRate = 0.01)
        {
            SamplesSet samplesSet = new SamplesSet();
            samplesSet.AddSample(sample);

            int iter = 0;
            while (true)
            {
                var error = parallel ? TrainingRunParallel(samplesSet) : TrainingRun(samplesSet);
                Console.WriteLine(error);
                iter++;

                if (error <= acceptableError) return iter;
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel, double learningRate = 0.01)
        {
            stopWatch.Restart();
            double error;
            int epoch = 0;
            while (true)
            {
                epoch++;
                error = parallel ? TrainingRunParallel(samplesSet) : TrainingRun(samplesSet);
                OnTrainProgress((epoch * 1.0) / epochsCount, error, stopWatch.Elapsed);

                if (epoch >= epochsCount || error <= acceptableError) break;
            }
            OnTrainProgress(1.0, error, stopWatch.Elapsed);
            stopWatch.Stop();
            return error;
        }

        protected override double[] Compute(double[] input)
        {
            return Predict(input).Last();
        }

        private readonly object lockObject = new object();

        private double TrainingRun(SamplesSet samples)
        {
            int batchCount = (samples.Count + batchSize - 1) / batchSize;

            double summaryError = 0.0;
            for (int i = 0; i < batchCount; i++)
            {
                var currentBatch = samples.samples.Skip(i * batchSize).Take(batchSize).ToArray();

                var outputs = new double[currentBatch.Length][][];

                for (int sampleIdx = 0; sampleIdx < currentBatch.Length; sampleIdx++)
                {
                    outputs[sampleIdx] = Predict(currentBatch[sampleIdx].input);
                    currentBatch[sampleIdx].ProcessPrediction(outputs[sampleIdx].Last());
                    summaryError += currentBatch[sampleIdx].EstimatedError();
                }

                var errors = Backpropagate(currentBatch, outputs);

                UpdateWeightsAndBiases(outputs, errors);
            }

            return summaryError;
        }
        private double TrainingRunParallel(SamplesSet samples)
        {
            int batchCount = (samples.Count + batchSize - 1) / batchSize;

            var batches = new Sample[batchCount][];

            for (int i = 0; i < batchCount; i++)
                batches[i] = samples.samples.Skip(i * batchSize).Take(batchSize).ToArray();

            for (int i = 0; i < batchCount; i++)
            {
                var currentBatch = batches[i];

                var outputs = new double[currentBatch.Length][][];

                Parallel.For(0, currentBatch.Length, sampleIdx =>
                {
                    outputs[sampleIdx] = PredictParallel(currentBatch[sampleIdx].input);
                    currentBatch[sampleIdx].ProcessPrediction(outputs[sampleIdx].Last());
                });

                var errors = BackpropagateParallel(currentBatch, outputs);

                UpdateWeightsAndBiasesParallel(outputs, errors);
            }

            double summaryError = 0.0;
            for (int i = 0; i < batchCount; i++)
                for (int sampleIdx = 0; sampleIdx < batches[i].Length; sampleIdx++)
                    summaryError += batches[i][sampleIdx].EstimatedError();

            return summaryError;
        }

        private double[][] PredictParallel(double[] input)
        {
            double[][] layersOutput = new double[layers.Count + 1][];
            layersOutput[0] = input;

            for (int layer = 0; layer < layers.Count; layer++)
            {
                layersOutput[layer + 1] = new double[layers[layer].biases.Length];

                Parallel.For(0, layers[layer].biases.Length, currNeuronIdx =>
                {
                    double sum = 0.0;

                    for (int prevNeuronIdx = 0; prevNeuronIdx < layers[layer].weights.GetLength(0); prevNeuronIdx++)
                        sum += layers[layer].weights[prevNeuronIdx, currNeuronIdx] * layersOutput[layer][prevNeuronIdx];

                    layersOutput[layer + 1][currNeuronIdx] = Activation(sum + layers[layer].biases[currNeuronIdx]);
                });
            }

            return layersOutput;
        }

        private double[][][] BackpropagateParallel(Sample[] currentBatch, double[][][] outputs)
        {
            var errors = new double[currentBatch.Length][][];
            Parallel.For(0, currentBatch.Length, sampleIdx =>
            {
                errors[sampleIdx] = new double[layers.Count][];

                errors[sampleIdx][layers.Count - 1] = new double[layers.Last().biases.Length];
                for (int currNeuronIdx = 0; currNeuronIdx < layers.Last().biases.Length; currNeuronIdx++)
                {
                    var expectedOutput = (int)currentBatch[sampleIdx].actualClass == currNeuronIdx ? 1 : 0;
                    var output = outputs[sampleIdx].Last()[currNeuronIdx];

                    errors[sampleIdx].Last()[currNeuronIdx] = (output - expectedOutput) * ActivationDerivative(output);
                }


                for (int layerIdx = layers.Count - 2; layerIdx >= 0; layerIdx--)
                {
                    errors[sampleIdx][layerIdx] = new double[layers[layerIdx].biases.Length];

                    Parallel.For(0, layers[layerIdx].biases.Length, prevNeuronIdx =>
                    {
                        double sum = 0.0;

                        for (int currNeuronIdx = 0; currNeuronIdx < layers[layerIdx + 1].weights.GetLength(1); currNeuronIdx++)
                            sum += layers[layerIdx + 1].weights[prevNeuronIdx, currNeuronIdx] * errors[sampleIdx][layerIdx + 1][currNeuronIdx];

                        errors[sampleIdx][layerIdx][prevNeuronIdx] = sum * ActivationDerivative(outputs[sampleIdx][layerIdx + 1][prevNeuronIdx]);
                    });

                }
            });
            return errors;
        }
        private void UpdateWeightsAndBiasesParallel(double[][][] outputs, double[][][] errors)
        {
            Parallel.For(0, layers.Count, layerIdx =>
            {
                var deltaLayer = new NNLayer
                {
                    weights = new double[layers[layerIdx].weights.GetLength(0), layers[layerIdx].weights.GetLength(1)],
                    biases = new double[layers[layerIdx].biases.Length]
                };

                Parallel.For(0, layers[layerIdx].biases.Length, nextNeuronIdx =>
                {

                    for (int sampleIdx = 0; sampleIdx < errors.Length; sampleIdx++)
                    {
                        deltaLayer.biases[nextNeuronIdx] += errors[sampleIdx][layerIdx][nextNeuronIdx];

                        for (int currNeuronIdx = 0; currNeuronIdx < layers[layerIdx].weights.GetLength(0); currNeuronIdx++)
                            deltaLayer.weights[currNeuronIdx, nextNeuronIdx] += errors[sampleIdx][layerIdx][nextNeuronIdx] * outputs[sampleIdx][layerIdx][currNeuronIdx];
                    }

                    layers[layerIdx].biases[nextNeuronIdx] -= learningRate * deltaLayer.biases[nextNeuronIdx];

                    for (int currNeuronIdx = 0; currNeuronIdx < layers[layerIdx].weights.GetLength(0); currNeuronIdx++)
                        layers[layerIdx].weights[currNeuronIdx, nextNeuronIdx] -= learningRate * deltaLayer.weights[currNeuronIdx, nextNeuronIdx];
                });
            });
        }

        private double[][] Predict(double[] input)
        {
            double[][] layersOutput = new double[layers.Count + 1][];
            layersOutput[0] = input;

            for (int layer = 0; layer < layers.Count; layer++)
            {
                layersOutput[layer + 1] = new double[layers[layer].biases.Length];

                for (int currNeuronIdx = 0; currNeuronIdx < layers[layer].biases.Length; currNeuronIdx++)
                {
                    double sum = 0.0;

                    for (int prevNeuronIdx = 0; prevNeuronIdx < layers[layer].weights.GetLength(0); prevNeuronIdx++)
                        sum += layers[layer].weights[prevNeuronIdx, currNeuronIdx] * layersOutput[layer][prevNeuronIdx];

                    layersOutput[layer + 1][currNeuronIdx] = Activation(sum + layers[layer].biases[currNeuronIdx]);
                }
            }

            return layersOutput;
        }

        private double[][][] Backpropagate(Sample[] currentBatch, double[][][] outputs)
        {
            var errors = new double[currentBatch.Length][][];
            for (int sampleIdx = 0; sampleIdx < currentBatch.Length; sampleIdx++)
            {
                errors[sampleIdx] = new double[layers.Count][];

                errors[sampleIdx][layers.Count - 1] = new double[layers.Last().biases.Length];
                for (int currNeuronIdx = 0; currNeuronIdx < layers.Last().biases.Length; currNeuronIdx++)
                {
                    var expectedOutput = (int)currentBatch[sampleIdx].actualClass == currNeuronIdx ? 1 : 0;
                    var output = outputs[sampleIdx].Last()[currNeuronIdx];

                    errors[sampleIdx].Last()[currNeuronIdx] = (output - expectedOutput) * ActivationDerivative(output);
                }

                for (int layerIdx = layers.Count - 2; layerIdx >= 0; layerIdx--)
                {
                    errors[sampleIdx][layerIdx] = new double[layers[layerIdx].biases.Length];
                    for (int prevNeuronIdx = 0; prevNeuronIdx < layers[layerIdx].biases.Length; prevNeuronIdx++)
                    {
                        double sum = 0.0;

                        for (int currNeuronIdx = 0; currNeuronIdx < layers[layerIdx + 1].weights.GetLength(1); currNeuronIdx++)
                            sum += layers[layerIdx + 1].weights[prevNeuronIdx, currNeuronIdx] * errors[sampleIdx][layerIdx + 1][currNeuronIdx];

                        errors[sampleIdx][layerIdx][prevNeuronIdx] = sum * ActivationDerivative(outputs[sampleIdx][layerIdx + 1][prevNeuronIdx]);
                    }

                }
            }
            return errors;
        }
        private void UpdateWeightsAndBiases(double[][][] outputs, double[][][] errors)
        {
            for (int layerIdx = 0; layerIdx < layers.Count; layerIdx++)
            {
                var deltaLayer = new NNLayer
                {
                    weights = new double[layers[layerIdx].weights.GetLength(0), layers[layerIdx].weights.GetLength(1)],
                    biases = new double[layers[layerIdx].biases.Length]
                };
                for (int nextNeuronIdx = 0; nextNeuronIdx < layers[layerIdx].biases.Length; nextNeuronIdx++)
                {

                    for (int sampleIdx = 0; sampleIdx < errors.Length; sampleIdx++)
                    {
                        deltaLayer.biases[nextNeuronIdx] += errors[sampleIdx][layerIdx][nextNeuronIdx];

                        for (int currNeuronIdx = 0; currNeuronIdx < layers[layerIdx].weights.GetLength(0); currNeuronIdx++)
                            deltaLayer.weights[currNeuronIdx, nextNeuronIdx] += errors[sampleIdx][layerIdx][nextNeuronIdx] * outputs[sampleIdx][layerIdx][currNeuronIdx];
                    }

                    layers[layerIdx].biases[nextNeuronIdx] -= learningRate * deltaLayer.biases[nextNeuronIdx];

                    for (int currNeuronIdx = 0; currNeuronIdx < layers[layerIdx].weights.GetLength(0); currNeuronIdx++)
                        layers[layerIdx].weights[currNeuronIdx, nextNeuronIdx] -= learningRate * deltaLayer.weights[currNeuronIdx, nextNeuronIdx];
                }
            }
        }
    }
}