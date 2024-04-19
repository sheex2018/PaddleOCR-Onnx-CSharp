using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PaddleOCRTestOnnx
{
    internal class TextClassifier
    {
        private InferenceSession _session;
        private List<string> input_names;
        private List<string> output_names;

        private int num_out;
        private int inpHeight = 48;
        private int inpWidth = 192;
        private List<float> input_image_;

        public TextClassifier(string modelpath)
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;
            _session = new InferenceSession(modelpath, sessionOptions);

            input_names = new List<string>();
            output_names = new List<string>();

            input_image_ = new List<float>();

            foreach (var name in this._session.InputMetadata.Keys)
            {
                this.input_names.Add(name);
            }

            foreach (var name in this._session.OutputMetadata.Keys)
            {
                this.output_names.Add(name);
            }

        }
       
        public int Predict(Mat cv_image)
        {
            Mat dstimg = this.Preprocess(cv_image);
            this.Normalize(dstimg);

            int[] input_shape_ = new int[] { 1, 3, this.inpHeight, this.inpWidth };

            var input_tensor_ = new DenseTensor<float>(input_image_.ToArray(), input_shape_);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>(input_names[0], input_tensor_)
            };

            var outputs = _session.Run(inputs);

            var pdata = outputs.First().AsTensor<float>().ToArray();

            int max_id = 0;
            float max_prob = -1;
            for (int i = 0; i < num_out; i++)
            {
                if (pdata[i] > max_prob)
                {
                    max_prob = pdata[i];
                    max_id = i;
                }
            }

            return max_id;
        }

        //是否可忽略
        private Mat Preprocess(Mat srcimg)
        {
            Mat dstimg = new Mat();
            int h = srcimg.Rows;
            int w = srcimg.Cols;
            float ratio = w / (float)h;
            int resized_w = (int)Math.Ceiling((float)this.inpHeight * ratio);
            if (Math.Ceiling(this.inpHeight * ratio) > this.inpWidth)
            {
                resized_w = this.inpWidth;
            }

            Cv2.Resize(srcimg, dstimg, new OpenCvSharp.Size(resized_w, this.inpHeight), 0, 0, InterpolationFlags.Linear);

            return dstimg;
        }

        public void Normalize(Mat img)
        {
            int row = img.Rows;
            int col = img.Cols;
            this.input_image_.Clear();
            this.input_image_.Capacity = this.inpHeight * this.inpWidth * img.Channels();
            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < row; i++)
                {
                    for (int j = 0; j < inpWidth; j++)
                    {
                        if (j < col)
                        {
                            float pix = img.Get<Vec3b>(i, j)[c];
                            this.input_image_.Add((pix / 255.0f - 0.5f) / 0.5f);
                        }
                        else
                        {
                            this.input_image_.Add(0);
                        }
                    }
                }
            }
        }
    }
}
