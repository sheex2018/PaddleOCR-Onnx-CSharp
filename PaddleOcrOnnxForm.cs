using OpenCvSharp;

namespace PaddleOCRTestOnnx
{
    public partial class PaddleOcrOnnxForm : Form
    {
        TextDetector _textDetector = null;
        TextClassifier _textClassifier = null;
        TextRecognizer _textRecognizer = null;
        public PaddleOcrOnnxForm()
        {
            InitializeComponent();
        }

        private void buttonImageExplorer_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.*|*.bmp;*.jpg;*.jpeg;*.tiff;*.tiff;*.png";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                textBoxImageFile.Text = ofd.FileName;
            }
        }

        private void buttonRecognitionInit_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.*|*.onnx";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                textBoxRecognitionFile.Text = ofd.FileName;
                _textRecognizer = new TextRecognizer(textBoxRecognitionFile.Text);
            }
        }

        private void buttonDetModelInit_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.*|*.onnx";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                textBoxDetModel.Text = ofd.FileName;

                _textDetector = new TextDetector(textBoxDetModel.Text);
            }
        }

        private void buttonClassifierInit_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "*.*|*.onnx";
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                textBoxClassifierFile.Text = ofd.FileName;

                _textClassifier = new TextClassifier(textBoxClassifierFile.Text);
            }
        }

        private void buttonDetModelInfer_Click(object sender, EventArgs e)
        {
            if (_textDetector == null)
                return;

            textBoxResults.Text = "";

            Mat srcImg = Cv2.ImRead(textBoxImageFile.Text);
            List<List<Point2f>> results = _textDetector.Detect(srcImg);

            for (int i = 0; i < results.Count; i++)
            {
                Mat textimg = _textDetector.GetRotateCropImage(srcImg, results[i].ToArray());

                if (_textClassifier != null)
                {
                    if (_textClassifier.Predict(textimg) == 1)
                    {
                        Cv2.Rotate(textimg, textimg, RotateFlags.Rotate90Clockwise);
                    }
                }

                textBoxResults.Text += _textRecognizer.PredictText(textimg) + System.Environment.NewLine;
            }

            //_textDetector.DrawPred(srcimg, results);

        }
    }
}