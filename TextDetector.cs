using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace PaddleOCRTestOnnx
{
    internal class TextDetector
    {
        private float binaryThreshold;
        private float polygonThreshold;
        private float unclipRatio;
        private int maxCandidates;
        private string modelPath;
        private SessionOptions sessionOptions;
        private InferenceSession _session;
        private List<string> inputNames;
        private List<string> outputNames;
        private List<float> inputImage;
        private float[] meanValues = { 0.485f, 0.456f, 0.406f };
        private float[] normValues = { 0.229f, 0.224f, 0.225f };

        private int shortSize = 736;
        private float longSideThresh = 80.0f;
        private float shortSideThresh = 3.0f;

        public TextDetector(string modelpath, SessionOptions opts = null)
        {
            this.binaryThreshold = 0.3f;
            this.polygonThreshold = 0.5f;
            this.unclipRatio = 1.6f;
            this.maxCandidates = 1000;

            this.modelPath = modelpath;
            this.sessionOptions = new SessionOptions();
            this.sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC;

            this._session = new InferenceSession(this.modelPath, this.sessionOptions);
            this.inputNames = new List<string>();
            this.outputNames = new List<string>();
            this.inputImage = new List<float>();

            //智能
            foreach (var name in this._session.InputMetadata.Keys)
            {
                this.inputNames.Add(name);
            }

            foreach (var name in this._session.OutputMetadata.Keys)
            {
                this.outputNames.Add(name);
            }
        }


        public List<List<Point2f>> Detect(Mat srcImg)
        {
            
            int h = srcImg.Rows;
            int w = srcImg.Cols;

            //0. 图像预处理 尺寸调整  归一化
            Mat dstImg = this.Preprocess(srcImg);
            this.Normalize(dstImg);
            
            //1. 构建输入张量
            int[] inputShape = { 1, 3, dstImg.Rows, dstImg.Cols };
            var inputTensor = new DenseTensor<float>(this.inputImage.ToArray(), inputShape);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(this.inputNames[0], inputTensor)
            };

            //2. 推理
            var outputs = this._session.Run(inputs);

            //3. 输出值解码
            var floatArray = outputs[0].AsTensor<float>().ToArray();
            int outputCount = 1;
            foreach (var dim in outputs[0].AsTensor<float>().Dimensions)
            {
                outputCount *= dim;
            }

            Mat binary = new Mat(dstImg.Rows, dstImg.Cols, MatType.CV_8UC1);
            Mat bitmap = new Mat(dstImg.Rows, dstImg.Cols, MatType.CV_8UC1);
            //Buffer.BlockCopy(floatArray, 0, binary.Data, 0, outputCount * sizeof(float));  //TODO <--看看高速
            for (int y = 0; y < dstImg.Rows; y++)
            {
                for (int x = 0; x < dstImg.Cols; x++)
                {
                    binary.Set<byte>(y, x, (byte)(255.0 * floatArray[y* dstImg.Cols+x]));  //网络输出值是0-1,所以乘以255
                    bitmap.Set<byte>(y, x, (byte)(floatArray[y * dstImg.Cols + x] > this.binaryThreshold ? 255:0));  //网络输出值是0-1,所以乘以255
                }
            }

            //Cv2.Threshold(binary, bitmap, this.binaryThreshold, 255, ThresholdTypes.Binary);
            //Mat bitmap = binary.Threshold(this.binaryThreshold * 255.0, 255, ThresholdTypes.Binary);

            float scaleHeight = (float)(h) / (float)(bitmap.Size(0));
            float scaleWidth = (float)(w) / (float)(bitmap.Size(1));

            //var contours = new List<Point[]>();
            OpenCvSharp.Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(bitmap, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);
            
            int numCandidate = Math.Min(contours.Length, this.maxCandidates > 0 ? this.maxCandidates : int.MaxValue);

            var confidences = new List<float>();
            var results = new List<List<Point2f>>();

            for (int i = 0; i < numCandidate; i++)
            {
                var contour = contours[i];

                // Calculate text contour score
                if (this.ContourScore(binary, contour) < this.polygonThreshold)
                    continue;

                // Rescale
                var contourScaled = new List<OpenCvSharp.Point>();
                contourScaled.AddRange(contour.Select(p => new OpenCvSharp.Point((int)(p.X * scaleWidth), (int)(p.Y * scaleHeight))));

                var box = Cv2.MinAreaRect(contourScaled);
                //float longSide = Math.Max(box.Size.Width, box.Size.Height);

                //if (longSide < this.longSideThresh)
                //    continue;
                float shortSide = Math.Min(box.Size.Width, box.Size.Height);
                if (shortSide < this.shortSideThresh)
                    continue;

                // minArea() rect is not normalized, it may return rectangles with angle=-90 or height < width
                bool swapSize = false;
                if (box.Size.Width < box.Size.Height || Math.Abs(box.Angle) >= 60.0f)                
                    swapSize = true;

                if (swapSize)
                {
                    float temp = box.Size.Width;
                    box.Size.Width = box.Size.Height;
                    box.Size.Height = temp;

                    if (box.Angle < 0)
                    {
                        box.Angle += 90;
                    }
                    else if (box.Angle > 0)
                    {
                        box.Angle -= 90;
                    }
                }

                var approx = new List<Point2f>();
                approx.AddRange(box.Points());

                var polygon = new List<Point2f>();
                this.Unclip(approx, polygon);

                box = Cv2.MinAreaRect(polygon);
                //longSide = Math.Max(box.Size.Width, box.Size.Height);
                //if (longSide < this.longSideThresh + 2)                
                //    continue;
                shortSide = Math.Min(box.Size.Width, box.Size.Height);
                if (shortSide < this.shortSideThresh+2.0f)
                    continue;

                results.Add(polygon);
            }

            //TODO 没有使用，忽略？
            confidences = Enumerable.Repeat(1.0f, contours.Length).ToList();

            return results;
        }

        public List<List<Point2f>> Detect(string imgPath)
        {
            Mat srcImg = Cv2.ImRead(imgPath);

            return Detect(srcImg);
        }
        //
        private Mat Preprocess(Mat srcImg)
        {
            Mat dstImg = new Mat();
            Cv2.CvtColor(srcImg, dstImg, ColorConversionCodes.BGR2RGB);

            int h = srcImg.Rows;
            int w = srcImg.Cols;
            float scaleH = 1;
            float scaleW = 1;

            if (h < w)
            {
                scaleH = (float)this.shortSize / (float)h;
                float tarW = (float)w * scaleH;
                tarW = tarW - (int)tarW % 32;
                tarW = Math.Max((float)32, tarW);
                scaleW = tarW / (float)w;
            }
            else
            {
                scaleW = (float)this.shortSize / (float)w;
                float tarH = (float)h * scaleW;
                tarH = tarH - (int)tarH % 32;
                tarH = Math.Max((float)32, tarH);
                scaleH = tarH / (float)h;
            }

            Cv2.Resize(dstImg, dstImg, new OpenCvSharp.Size((int)(scaleW * dstImg.Cols), (int)(scaleH * dstImg.Rows)), interpolation: InterpolationFlags.Linear);
            return dstImg;
        }

        private void Normalize(Mat img)
        {
            int row = img.Rows;
            int col = img.Cols;
            this.inputImage.Clear();

            for (int c = 0; c < 3; c++)
            {
                for (int i = 0; i < row; i++)
                {
                    for (int j = 0; j < col; j++)
                    {
                        float pix = img.Get<Vec3b>(i, j)[c];
                        this.inputImage.Add((pix / 255.0f - this.meanValues[c]) / this.normValues[c]);
                    }
                }
            }
        }

        //    public List<List<Point2f>> OrderPointsClockwise(List<List<Point2f>> results)
        //    {
        //        var orderPoints = new List<List<Point2f>>(results);

        //        for (int i = 0; i < results.Count; i++)
        //        {
        //            float maxSumPts = -10000;
        //            float minSumPts = 10000;
        //            float maxDiffPts = -10000;
        //            float minDiffPts = 10000;

        //            int maxSumPtsId = 0;
        //            int minSumPtsId = 0;
        //            int maxDiffPtsId = 0;
        //            int minDiffPtsId = 0;

        //            for (int j = 0; j < 4; j++)
        //            {
        //                float sumPt = results[i][j].X + results[i][j].Y;

        //                if (sumPt > maxSumPts)
        //                {
        //                    maxSumPts = sumPt;
        //                    maxSumPtsId = j;
        //                }

        //                if (sumPt < minSumPts)
        //                {
        //                    minSumPts = sumPt;
        //                    minSumPtsId = j;
        //                }

        //                float diffPt = results[i][j].Y - results[i][j].X;

        //                if (diffPt > maxDiffPts)
        //                {
        //                    maxDiffPts = diffPt;
        //                    maxDiffPtsId = j;
        //                }

        //                if (diffPt < minDiffPts)
        //                {
        //                    minDiffPts = diffPt;
        //                    minDiffPtsId = j;
        //                }
        //            }

        //            orderPoints[i][0].X = results[i][minSumPtsId].X;
        //            orderPoints[i][0].Y = results[i][minSumPtsId].Y;
        //            orderPoints[i][2].X = results[i][maxSumPtsId].X;
        //            orderPoints[i][2].Y = results[i][maxSumPtsId].Y;

        //            orderPoints[i][1].X = results[i][minDiffPtsId].X;
        //            orderPoints[i][1].Y = results[i][minDiffPtsId].Y;
        //            orderPoints[i][3].X = results[i][maxDiffPtsId].X;
        //            orderPoints[i][3].Y = results[i][maxDiffPtsId].Y;
        //        }

        //        return orderPoints;
        //    }

        //    public void DrawPred(Mat srcImg, List<List<Point2f>> results)
        //    {
        //        for (int i = 0; i < results.Count; i++)
        //        {
        //            for (int j = 0; j < 4; j++)
        //            {
        //                Cv2.Circle(srcImg, new Point((int)results[i][j].X, (int)results[i][j].Y), 2, Scalar.Red, -1);

        //                if (j < 3)
        //                {
        //                    Cv2.Line(srcImg, new Point((int)results[i][j].X, (int)results[i][j].Y), new Point((int)results[i][j + 1].X, (int)results[i][j + 1].Y), Scalar.Green);
        //                }
        //                else
        //                {
        //                    Cv2.Line(srcImg, new Point((int)results[i][j].X, (int)results[i][j].Y), new Point((int)results[i][0].X, (int)results[i][0].Y), Scalar.Green);
        //                }
        //            }
        //        }
        //    }

        //计算轮廓分值 20240416未完全理解
        // TODO 注意返回值的归一化
        private float ContourScore(Mat binary, OpenCvSharp.Point[] contour)
        {
            //1. 获取轮廓点的外接矩形
            Rect rect = Cv2.BoundingRect(contour);
            int xmin = Math.Max(rect.X, 0);
            int xmax = Math.Min(rect.X + rect.Width, binary.Cols - 1);
            int ymin = Math.Max(rect.Y, 0);
            int ymax = Math.Min(rect.Y + rect.Height, binary.Rows - 1);

            //2. 填充外接矩形内，由轮廓点围成的多边形
            Mat binROI = new Mat(binary, new Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
            Mat mask = Mat.Zeros(new OpenCvSharp.Size(xmax - xmin + 1, ymax - ymin + 1), MatType.CV_8U);
            var roiContour = contour.Select(p => new OpenCvSharp.Point(p.X - xmin, p.Y - ymin)).ToList();
            Cv2.FillPoly(mask, new List<List<OpenCvSharp.Point>> { roiContour }, 1); // 1
            
            //3. 计算填充多边形区域的均值 

            Scalar mean = Cv2.Mean(binROI, mask);

            return (float)mean.Val0/255.0f;

        }

        // 未理解该函数的意义 20240416
        // 或许可参考 DBNet后处理unclip()函数转C++  https://www.jianshu.com/p/0227c40b0736  
        private void Unclip(List<Point2f> inPoly, List<Point2f> outPoly)
        {
            float area = (float)Cv2.ContourArea(inPoly);    //轮廓面积
            float length = (float)Cv2.ArcLength(inPoly, true); //轮廓周长
            float distance = area * this.unclipRatio / length;

            int numPoints = inPoly.Count;
            var newLines = new List<List<Point2f>>();

            for (int i = 0; i < numPoints; i++)
            {
                var newLine = new List<Point2f>();
                Point2f pt1 = inPoly[i];
                Point2f pt2 = inPoly[(i - 1 + numPoints) % numPoints];
                Point2f vec = pt1 - pt2;
                float unclipDis = (float)(distance / Math.Sqrt(vec.X * vec.X + vec.Y * vec.Y));
                
                Point2f rotateVec = new Point2f(vec.Y * unclipDis, -vec.X * unclipDis);
                newLine.Add(new Point2f(pt1.X + rotateVec.X, pt1.Y + rotateVec.Y));
                newLine.Add(new Point2f(pt2.X + rotateVec.X, pt2.Y + rotateVec.Y));
                newLines.Add(newLine);
            }

            int numLines = newLines.Count;

            for (int i = 0; i < numLines; i++)
            {
                Point2f a = newLines[i][0];
                Point2f b = newLines[i][1];
                Point2f c = newLines[(i + 1) % numLines][0];
                Point2f d = newLines[(i + 1) % numLines][1];
                Point2f pt;
                Point2f v1 = b - a;
                Point2f v2 = d - c;
                float cosAngle = (float)((v1.X * v2.X + v1.Y * v2.Y) / (Math.Sqrt(v1.X * v1.X + v1.Y * v1.Y) * Math.Sqrt(v2.X * v2.X + v2.Y * v2.Y)));

                if (Math.Abs(cosAngle) > 0.7)
                {
                    pt.X = (b.X + c.X) * 0.5f;
                    pt.Y = (b.Y + c.Y) * 0.5f;
                }
                else
                {
                    float denom = a.X * (float)(d.Y - c.Y) + b.X * (float)(c.Y - d.Y) +
                        d.X * (float)(b.Y - a.Y) + c.X * (float)(a.Y - b.Y);
                    float num = a.X * (float)(d.Y - c.Y) + c.X * (float)(a.Y - d.Y) + d.X * (float)(c.Y - a.Y);
                    float s = num / denom;

                    pt.X = a.X + s * (b.X - a.X);
                    pt.Y = a.Y + s * (b.Y - a.Y);
                }

                outPoly.Add(pt);
            }
        }

        //基于vertices围成的外接矩形，似乎没有作用
        public Mat GetRotateCropImage(Mat frame, Point2f[] vertices)
        {
            Rect rect = Cv2.BoundingRect(vertices);
            Mat cropImg = new Mat(frame, rect);
            OpenCvSharp.Size outputSize = new OpenCvSharp.Size(rect.Width, rect.Height);

            List<Point2f> targetVertices = new List<Point2f>
            {
                new Point2f(0, outputSize.Height),
                new Point2f(0, 0),
                new Point2f(outputSize.Width, 0),
                new Point2f(outputSize.Width, outputSize.Height)
            };

            for (int i = 0; i < 4; i++)
            {
                vertices[i].X -= rect.X;
                vertices[i].Y -= rect.Y;
            }

            Mat rotationMatrix = Cv2.GetPerspectiveTransform(vertices.ToArray(), targetVertices.ToArray());
            Mat result = new Mat();
            Cv2.WarpPerspective(cropImg, result, rotationMatrix, outputSize, borderMode: BorderTypes.Replicate);

            return result;
        }
    }

}
