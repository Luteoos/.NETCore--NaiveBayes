using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using NaiveBayes;

namespace BayesNaive
{
    class Program
    {
        private static NaiveBayes _nb;

        static void Main(string[] args)
        {
            _nb=new NaiveBayes();
            if (args!=null)
            {
                string tempData;
                using (StreamReader sr=new StreamReader(args[0]))
                {
                     tempData = sr.ReadLine();
                }
                _nb._trainSetPath = args[1];
                _nb.Run(tempData);
            }
            else
            {
                Console.WriteLine("Wrong arguments given!");
            }
        }
    }

    class NaiveBayes
    {
        private DataTable _input;
        private List<Document> _trainData;
        private List<string> _allOptions;
        public string _trainSetPath;

        public void Run(string inputPreferences)
        {
            _allOptions=new List<string>();
            string pickedbeer = "";
            double result = 0;
            _trainData =new List<Document>();

            FillTrainData();
            Classifier classify=new Classifier(_trainData);
            foreach (var a in _allOptions)
            {
                if (classify.Probability(a, inputPreferences) >= result)
                {
                    result = classify.Probability(a, inputPreferences);
                    pickedbeer = a;
                }
            }
            Console.WriteLine(pickedbeer+", Probability="+result);
            Console.Read();
        }

        private void FillTrainData()
        {
            using (StreamReader sr = new StreamReader(_trainSetPath))
            {
                while (!sr.EndOfStream)
                {
                    string[] tempArray;
                    string tempdata;

                   tempArray = sr.ReadLine().Split(',');
                   tempdata = string.Join(',', tempArray[1], tempArray[2], tempArray[3], tempArray[4],
                        tempArray[0]);
                    _allOptions.Add(tempArray[5]);
                    _trainData.Add(new Document(tempArray[5],tempdata));
                }
            }
        }
    }

}

