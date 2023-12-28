using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using AIMLTGBot;

namespace NeuralNetwork1
{
    public class DataSet
    {

        static Random rand = new Random();
        private static Dictionary<string, FigureType> Classes = new Dictionary<string, FigureType>()
        {
            {"Happy", FigureType.Happy }, {"Angry", FigureType.Angry}, {"Sad", FigureType.Sad}, {"Tongue", FigureType.Tongue},
            {"PokerFace", FigureType.PokerFace}, {"Smile", FigureType.Smile}, {"Surprised", FigureType.Surprised}, {"Wink", FigureType.Wink}
        };
        static Sample bitmapToSample(Bitmap processed)
        {
            Rectangle rect = new Rectangle(0, 0, processed.Width, processed.Height);
            BitmapData bmpData = processed.LockBits(rect, ImageLockMode.ReadOnly, processed.PixelFormat);
            double[] input = new double[48 * 48];
            unsafe
            {
                byte* ptr = (byte*)bmpData.Scan0;
                int heightInPixels = bmpData.Height;
                int widthInBytes = bmpData.Stride;
                for (int y = 0; y < heightInPixels; y++)
                {
                    for (int x = 0; x < widthInBytes; x = x + 1)
                    {
                        double grayValue = ptr[(y * bmpData.Stride) + x] / 255.0;
                        input[(y * bmpData.Stride) + x] = grayValue;

                    }
                }
            }
            Sample sample = new Sample(input, 8);
            return sample;
        }
        public static SamplesSet GetDataSet(string path = "../../mydataset1/")
        {
            SamplesSet s = new SamplesSet();
            void LoadClass(string subdir)
            {
                foreach (var file in Directory.GetFiles(Path.Combine(path, subdir)))
                {
                    // if (rand.Next(0, 100) > 10) continue;
                    Bitmap f = new Bitmap(file);

                   s.AddSample(bitmapToSample(f));
                }
            }

            foreach (var subdir in Directory.GetDirectories(path))
            {
                LoadClass(Path.GetFileName(subdir));
            }

            s.samples = s.samples.OrderBy(i => rand.Next()).ToList();
            
            return s;
        }
    }
}
