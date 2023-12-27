using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        double[][,] weight;
        double[][] layers;
        double[][] errors;
        double step = 0.15;

        bool parallel = false;

        Stopwatch stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            this.layers = new double[structure.Length][];
            this.layers[0] = new double[structure[0] + 1]; // +1 bios
            this.layers[0][structure[0]] = 1; // bios

            this.errors = new double[structure.Length][];
            //this.errors[0] = new float[layers[0] + 1];

            this.weight = new double[structure.Length - 1][,];
            for (UInt16 i = 1; i < structure.Length; i++)
            {
                this.layers[i] = new double[structure[i] + (structure.Length - 1 == i ? 0 : 1)]; // +1 bios, but a last layer
                this.errors[i] = new double[structure[i]];
                if (structure.Length - 1 != i) this.layers[i][structure[i]] = 1; // bios
                this.weight[i - 1] = new double[structure[i - 1] + 1, structure[i]]; // +1 bios               
            }
            Randomize();
        }

        private void Randomize()
        {
            Random r = new Random();
            if (parallel)
            {
                Parallel.ForEach(this.weight, w => {
                    for (int i = 0; i < w.GetLength(0); i++)
                    {
                        for (int j = 0; j < w.GetLength(1); j++)
                        {
                            w[i, j] = r.NextDouble() * 2 - 1;
                        }
                    }
                });
            }
            else
            {
                foreach (var w in weight)
                {
                    for (int i = 0; i < w.GetLength(0); i++)
                    {
                        for (int j = 0; j < w.GetLength(1); j++)
                        {
                            w[i, j] = r.NextDouble() * 2 - 1;
                        }
                    }
                }         
            }
        }

        private void PushForward()
        {
            if (parallel)
            {
                for (int k = 1; k < layers.Length; k++)
                {
                    Parallel.For(0, weight[k - 1].GetLength(1), j =>
                    {
                        layers[k][j] = 0;
                        for (int i = 0; i < weight[k - 1].GetLength(0); i++) //layers[k-1].Length
                        {
                            layers[k][j] += weight[k - 1][i, j] * layers[k - 1][i];
                        }
                        layers[k][j] = Sigmoid(layers[k][j]);
                    });
                }
            }
            else
            {
                for (int k = 1; k < layers.Length; k++)
                {
                    for (int j = 0; j < weight[k - 1].GetLength(1); j++) //layers[k].Length
                    {
                        layers[k][j] = 0;
                        for (int i = 0; i < weight[k - 1].GetLength(0); i++) //layers[k-1].Length
                        {
                            layers[k][j] += weight[k - 1][i, j] * layers[k - 1][i];
                        }
                        layers[k][j] = Sigmoid(layers[k][j]);
                    }
                }
            }
        }

        private double Sigmoid(double value)
        {
            return 1f / (1f + Math.Exp(-value));
        }

        private void BackPropagation(uint r_i)
        {
            if (parallel)
            {
                // last layer
                int k = layers.Length - 1;
                for (int j = 0; j < layers[k].Length; j++)
                {
                    errors[k][j] = -(layers[k][j] * (1f - layers[k][j])) * ((r_i == j ? 1f : 0f) - layers[k][j]);
                }

                // inner layers
                for (k = layers.Length - 2; k > 0; k--)
                {
                    Parallel.For(0, layers[k].Length - 1, i =>
                    {
                        errors[k][i] = 0f;
                        for (int j = 0; j < weight[k].GetLength(1); j++)
                        {
                            errors[k][i] += weight[k][i, j] * errors[k + 1][j];
                        }
                        errors[k][i] *= (layers[k][i] * (1f - layers[k][i]));
                    });

                }

                // tuning
                for (k = 0; k < weight.Length; k++)
                {
                    Parallel.For(0, weight[k].GetLength(1), j =>
                    {
                        for (int i = 0; i < weight[k].GetLength(0); i++)
                        {
                            weight[k][i, j] += -step * layers[k][i] * errors[k + 1][j];
                        }
                    });
                }
            }
            else
            {
                // last layer
                int k = layers.Length - 1;
                for (int j = 0; j < layers[k].Length; j++)
                {
                    errors[k][j] = -(layers[k][j] * (1f - layers[k][j])) * ((r_i == j ? 1f : 0f) - layers[k][j]);
                }

                // inner layers
                for (k = layers.Length - 2; k > 0; k--)
                {
                    for (int i = 0; i < layers[k].Length - 1; i++)
                    {
                        errors[k][i] = 0f;
                        for (int j = 0; j < weight[k].GetLength(1); j++)
                        {
                            errors[k][i] += weight[k][i, j] * errors[k + 1][j];
                        }
                        errors[k][i] *= (layers[k][i] * (1f - layers[k][i]));
                    }
                }

                // tuning
                for (k = 0; k < weight.Length; k++)
                {
                    for (int j = 0; j < weight[k].GetLength(1); j++)
                    {
                        for (int i = 0; i < weight[k].GetLength(0); i++)
                        {
                            weight[k][i, j] += -step * layers[k][i] * errors[k + 1][j];
                        }
                    }
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            this.parallel = parallel;
            int ret = 0;
            double estimatedError = 0;
            
            stopWatch.Restart();

            while(true){
                if(Predict(sample) == sample.actualClass && 
                    (estimatedError = sample.EstimatedError()) < acceptableError)
                {
                    OnTrainProgress(1.0, estimatedError, stopWatch.Elapsed);
                    stopWatch.Stop();
                    return ret;
                }
                else
                {
                    ret++;
                    BackPropagation((uint)sample.actualClass);
                }
                OnTrainProgress(Math.Min(estimatedError / acceptableError, 1.0), estimatedError, stopWatch.Elapsed);
            }         
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            this.parallel = parallel;
            double estimatedError = 0;

            stopWatch.Restart();
            
            for (int i = 0; i < epochsCount; i++)
            {
                estimatedError = 0;
                foreach (var sample in samplesSet.samples)
                {
                    Predict(sample);
                    estimatedError += sample.EstimatedError();             
                    BackPropagation((uint)sample.actualClass);         
                }
                OnTrainProgress((i * 1.0) / epochsCount, estimatedError / samplesSet.Count, stopWatch.Elapsed);
                if (estimatedError / samplesSet.Count < acceptableError)
                {
                    OnTrainProgress(1.0, estimatedError / samplesSet.Count, stopWatch.Elapsed);
                    stopWatch.Stop();
                    return estimatedError / samplesSet.Count;
                }
            }
            OnTrainProgress(1.0, estimatedError / samplesSet.Count, stopWatch.Elapsed);
            stopWatch.Stop();
            return estimatedError / samplesSet.Count;
        }

        protected override double[] Compute(double[] input)
        {
            for (int i = 0; i < input.Length; i++) layers[0][i] = input[i];
            //layers[0] = input;
            PushForward();
            return layers.Last();
        }
    }
}