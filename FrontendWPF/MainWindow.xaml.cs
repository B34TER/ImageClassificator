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
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            UploadingFilesList.Items.Clear();
            OpenFileDialog openFileDialog = new OpenFileDialog() { Multiselect = false };
            bool? response = openFileDialog.ShowDialog();
            if (response == true)
            {
                // Get Selected Files
                string[] files = openFileDialog.FileNames;

                // Iterate and add all selected files to upload
                for (int i = 0; i < files.Length; i++)
                {
                    string filename = Path.GetFileName(files[i]);
                    FileInfo fileInfo = new FileInfo(files[i]);
                    UploadingFilesList.Items.Add(new fileDetail()
                    {
                        FileName = filename,
                        // Converting bytes to Mb => bytes / 1.049e+6
                        FileSize = string.Format("{0} {1}", (fileInfo.Length / 1.049e+6).ToString("0.0"), "Mb"),
                        UploadProgress = 100

                    });
                }

                ImageSource.Source = new BitmapImage(new Uri(files[0], UriKind.Absolute));
                var predictionResult = Backend.Predictor.MakePrediction(files[0]);
                PredictionLabel.Text = $"It is: {predictionResult.PredictedLabelValue}";
                PredictionScore.Text = $"Prediction score: {predictionResult.Score[0]}";
            }
        }

        private void Rectangle_Drop(object sender, DragEventArgs e)
        {
            UploadingFilesList.Items.Clear();
            // Checking what kind of file is user dropping
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);

                // Iterate and add all selected files to upload
                for (int i = 0; i < files.Length; i++)
                {
                    string filename = Path.GetFileName(files[i]);
                    FileInfo fileInfo = new FileInfo(files[i]);
                    UploadingFilesList.Items.Add(new fileDetail()
                    {
                        FileName = filename,
                        // Converting bytes to Mb => bytes / 1.049e+6
                        FileSize = string.Format("{0} {1}", (fileInfo.Length / 1.049e+6).ToString("0.0"), "Mb"),
                        UploadProgress = 100

                    });
                }

                ImageSource.Source = new BitmapImage(new Uri(files[0], UriKind.Absolute));
                var predictionResult = Backend.Predictor.MakePrediction(files[0]);
                PredictionLabel.Text = $"It is: {predictionResult.PredictedLabelValue}";
                PredictionScore.Text = $"Prediction score - {predictionResult.Score[0]}";
            }
        }
    }
}
