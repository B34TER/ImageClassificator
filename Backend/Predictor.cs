using Microsoft.ML;

namespace Backend
{
    public static class Predictor
    {
        static readonly string workingDirectory = Environment.CurrentDirectory;
        static readonly string solutionDirectory = Directory.GetParent(workingDirectory).Parent.Parent.Parent.FullName;
        static readonly string _assetsPath = Path.Combine(solutionDirectory, "assets");
        static readonly string _imagesFolder = Path.Combine(_assetsPath, "images");
        static readonly string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
        static readonly string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

        public static ITransformer GenerateModel(MLContext mlContext)
        {
            IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, inputColumnName: nameof(ImageData.ImagePath))
                    // The image transforms transform the images into the model's expected format.
                    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                    .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                        ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "softmax2_pre_activation"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                    .AppendCacheCheckpoint(mlContext);

            IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

            ITransformer model = pipeline.Fit(trainingData);

            return model;
        }
        public static ImagePrediction ClassifySingleImage(MLContext mlContext, ITransformer model, string imagePath)
        {
            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };

            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            return new ImagePrediction()
            {
                PredictedLabelValue = prediction.PredictedLabelValue,
                Score = new float[] { prediction.Score.Max() }
            };
        }

        public static ImagePrediction MakePrediction(string imagePath)
        {
            MLContext mlContext = new MLContext();

            ITransformer model = GenerateModel(mlContext);

            return ClassifySingleImage(mlContext, model, imagePath);
        }
    }
}
