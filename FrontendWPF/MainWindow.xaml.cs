using Backend;
using ImageClassificationWPF.CustomControl;
using Microsoft.Win32;
using System;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;

namespace FrontendWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly Predictor _predictor;

        public MainWindow()
        {
            InitializeComponent();

            _predictor = new Predictor();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            UploadingFilesList.Items.Clear();
            OpenFileDialog openFileDialog = new OpenFileDialog() { Multiselect = false };
            bool? response = openFileDialog.ShowDialog();
            if (response == true)
            {
                // Get Selected Files
                string file = openFileDialog.FileName;
                string filename = Path.GetFileName(file);
                FileInfo fileInfo = new FileInfo(file);

                AddItemToFilesList(filename, fileInfo);

                var predictionResult = _predictor.ClassifySingleImage(file);
                ImageSource.Source = new BitmapImage(new Uri(file, UriKind.Absolute));
                PredictionLabel.Text = $"It is - {predictionResult.PredictedLabelValue}";
                PredictionScore.Text = $"Prediction score - {predictionResult.Score[0]}";
            }
        }

        private void AddItemToFilesList(string filename, FileInfo fileInfo)
        {
            UploadingFilesList.Items.Add(new fileDetail()
            {
                FileName = filename,
                // Converting bytes to Mb => bytes / 1.049e+6
                FileSize = string.Format("{0} {1}", (fileInfo.Length / 1.049e+6).ToString("0.0"), "Mb"),
                UploadProgress = 100
            });
        }

        private void Rectangle_Drop(object sender, DragEventArgs e)
        {
            UploadingFilesList.Items.Clear();
            // Checking what kind of file is user dropping
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                var files = (string[])e.Data.GetData(DataFormats.FileDrop);
                string file = files[0];
                string filename = Path.GetFileName(file);
                FileInfo fileInfo = new FileInfo(file);

                AddItemToFilesList(filename, fileInfo);

                var predictionResult = _predictor.ClassifySingleImage(file);
                ImageSource.Source = new BitmapImage(new Uri(file, UriKind.Absolute));
                PredictionLabel.Text = $"It is - {predictionResult.PredictedLabelValue}";
                PredictionScore.Text = $"Prediction score - {predictionResult.Score[0]}";
            }
        }
    }
}
