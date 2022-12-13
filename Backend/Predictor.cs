using Microsoft.ML;

namespace Backend
{
    public class Predictor
    {
        private string _assetsPath;
        private string _imagesFolder;
        private string _trainTagsTsv;
        private string _inceptionTensorFlowModel;
        private readonly ITransformer _model;
        private readonly MLContext _mlContext;

        public Predictor()
        {
            string workingDirectory = Environment.CurrentDirectory;
            string solutionDirectory = Directory.GetParent(workingDirectory).Parent.Parent.Parent.FullName;
            _assetsPath = Path.Combine(solutionDirectory, "assets");
            _imagesFolder = Path.Combine(_assetsPath, "images");
            _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
            _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

            _mlContext = new MLContext();
            _model = GenerateModel(_mlContext);
        }

        private ITransformer GenerateModel(MLContext mlContext)
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
        public ImagePrediction ClassifySingleImage(string imagePath)
        {
            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };

            var predictor = _mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(_model);
            var prediction = predictor.Predict(imageData);

            return new ImagePrediction()
            {
                PredictedLabelValue = prediction.PredictedLabelValue,
                Score = new float[] { prediction.Score.Max() }
            };
        }
    }
}
