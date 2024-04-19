using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PaddleOCRTestOnnx
{
    internal class DBPostProcess
    {
        private int _min_size;
        private double _thresh;
        private double _box_thresh;
        private int _max_candidates;
        private double _unclip_ratio;

        public DBPostProcess(double thresh = 0.3, double box_thresh = 0.5, int max_candidates = 500, double unclip_ratio = 1.6)
        {
            this._min_size = 3;
            this._thresh = thresh;
            this._box_thresh = box_thresh;
            this._max_candidates = max_candidates;
            this._unclip_ratio = unclip_ratio;
        }

        public (List<List<Point2f>>, List<double>) Process(int width, int height, Mat bitmap, bool is_output_polygon = false)
        {
            //二值化图
            Mat segmentation = bitmap.Threshold(this._thresh, 1.0, ThresholdTypes.Binary);

            List<List<Point2f>> boxes = new List<List<Point2f>>();
            List<double> scores = new List<double>();

            if (is_output_polygon)
            {
                //(List<List<Point2f>> polygons, List<double> polygonScores) = PolygonsFromBitmap(bitmap, segmentation, width, height);
                //boxes.AddRange(polygons);
                //scores.AddRange(polygonScores);
            }
            else
            {
                (List<List<Point2f>> boxList, List<double> boxScores) = BoxesFromBitmap(bitmap, segmentation, width, height);
                boxes.AddRange(boxList);
                scores.AddRange(boxScores);
            }

            return (boxes, scores);
        }


        //private (List<List<Point2f>>, List<double>) PolygonsFromBitmap(Mat pred, Mat bitmap, int dest_width, int dest_height)
        //{
        //    List<List<Point2f>> boxes = new List<List<Point2f>>();
        //    List<double> scores = new List<double>();
        //    int height = bitmap.Rows;
        //    int width = bitmap.Cols;

        //    Point[][] contours;
        //    HierarchyIndex[] hierarchy;
        //    Cv2.FindContours(bitmap, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

        //    foreach (Point[] contour in contours.Take(max_candidates))
        //    {
        //        double epsilon = 0.005 * Cv2.ArcLength(contour, true);
        //        Point2f[] approx = Cv2.ApproxPolyDP(contour, epsilon, true);
        //        Point2f[] points = approx;
        //        if (points.Length < 4)
        //        {
        //            continue;
        //        }

        //        double score = BoxScoreFast(pred, contour.Select(p => new Point2f(p.X, p.Y)).ToArray());
        //        if (box_thresh > score)
        //        {
        //            continue;
        //        }

        //        if (points.Length > 2)
        //        {
        //            List<Point2f> box = Unclip(points.ToList(), unclip_ratio);
        //            if (box.Count > 1)
        //            {
        //                continue;
        //            }
        //        }
        //        else
        //        {
        //            continue;
        //        }

        //        Point2f[] boxPoints = box.Select(p => new Point2f(p.X, p.Y)).ToArray();
        //        (_, double sside) = GetMiniBoxes(boxPoints);
        //        if (sside < min_size + 2)
        //        {
        //            continue;
        //        }

        //        if (!(dest_width is int))
        //        {
        //            dest_width = dest_width.Item();
        //            dest_height = dest_height.Item();
        //        }

        //        boxPoints = boxPoints.Select(p => new Point2f(Math.Clamp(Math.Round(p.X / width * dest_width), 0, dest_width), Math.Clamp(Math.Round(p.Y / height * dest_height), 0, dest_height))).ToArray();
        //        boxes.Add(boxPoints.ToList());
        //        scores.Add(score);
        //    }

        //    return (boxes, scores);
        //}

        private (List<List<Point2f>>, List<double>) BoxesFromBitmap(Mat bitmap, Mat segmentaton, int dest_width, int dest_height)
        {
            List<List<Point2f>> boxes = new List<List<Point2f>>();
            List<double> scores = new List<double>();

            int height = segmentaton.Rows;
            int width = segmentaton.Cols;

            OpenCvSharp.Point[][] contours;  
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(segmentaton, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            int num_contours = Math.Min(contours.Length, this._max_candidates);

            Point2f[][] boxArray = new Point2f[num_contours][];
            double[] scoreArray = new double[num_contours];
            for (int index = 0; index < num_contours; index++)
            {
                OpenCvSharp.Point[] contour = contours[index];
                List<Point2f> points;
                double sside;
                (points, sside) = GetMiniBoxes(contour.Select(p => new Point2f(p.X, p.Y)).ToArray());
                if (sside < this._min_size)
                    continue;
                
                //
                //double score = BoxScoreFast(bitmap, contour.Select(p => new Point2f(p.X, p.Y)).ToArray());
                double score = BoxScoreFast(bitmap, contour);
                if ( score < this._box_thresh)
                    continue;


                //Point2f[] box = Unclip(points.ToList(), unclip_ratio).ToArray();
                //(_, sside) = GetMiniBoxes(box);
                //if (sside < min_size + 2)
                //{
                //    continue;
                //}

                //if (!(dest_width is int))
                //{
                //    dest_width = dest_width.Item();
                //    dest_height = dest_height.Item();
                //}

                //box = box.Select(p => new Point2f(Math.Clamp(Math.Round(p.X / width * dest_width), 0, dest_width), Math.Clamp(Math.Round(p.Y / height * dest_height), 0, dest_height))).ToArray();
                //boxArray[index] = box;
                //scoreArray[index] = score;
            }

            boxes.AddRange(boxArray.Select(box => box.ToList()));
            scores.AddRange(scoreArray);

            return (boxes, scores);
        }

        //private List<Point2f> Unclip(List<Point2f> box, double unclip_ratio = 1.5)
        //{
        //    Polygon poly = new Polygon(box.Select(p => new IntPoint((long)p.X, (long)p.Y)).ToList());
        //    double distance = poly.Area * unclip_ratio / poly.Perimeter;
        //    ClipperOffset offset = new ClipperOffset();
        //    offset.AddPath(poly, JoinType.jtRound, EndType.etClosedPolygon);
        //    List<List<IntPoint>> expanded = new List<List<IntPoint>>();
        //    offset.Execute(ref expanded, distance);
        //    return expanded.SelectMany(p => p).Select(p => new Point2f((float)p.X, (float)p.Y)).ToList();
        //}

        private (List<Point2f>, double) GetMiniBoxes(Point2f[] contour)
        {
            RotatedRect boundingBox = Cv2.MinAreaRect(contour);
            List<Point2f> points= boundingBox.Points().OrderBy(p => p.X).ToList(); 

            int index_1, index_2, index_3, index_4;
            if (points[1].Y > points[0].Y)
            {
                index_1 = 0;
                index_4 = 1;
            }
            else
            {
                index_1 = 1;
                index_4 = 0;
            }

            if (points[3].Y > points[2].Y)
            {
                index_2 = 2;
                index_3 = 3;
            }
            else
            {
                index_2 = 3;
                index_3 = 2;
            }

            List<Point2f> box = new List<Point2f> { points[index_1], points[index_2], points[index_3], points[index_4] };
            return (box, Math.Min(boundingBox.Size.Width, boundingBox.Size.Height));
        }

        private double BoxScoreFast(Mat bitmap, OpenCvSharp.Point[] box)
        {
            int h = bitmap.Rows;
            int w = bitmap.Cols;

            OpenCvSharp.Point[] _box=new OpenCvSharp.Point[box.Length];
            box.CopyTo(_box,0);

            double xmin = Math.Clamp(Math.Floor((double)_box.Select(p => p.X).Min()), 0, w - 1);
            double xmax = Math.Clamp(Math.Ceiling((double)_box.Select(p => p.X).Max()), 0, w - 1);
            double ymin = Math.Clamp(Math.Floor((double)_box.Select(p => p.Y).Min()), 0, h - 1);
            double ymax = Math.Clamp(Math.Ceiling((double)_box.Select(p => p.Y).Max()), 0, h - 1);

            Mat mask = new Mat((int)(ymax - ymin + 1), (int)(xmax - xmin + 1), MatType.CV_8UC1, Scalar.Black);
            for (int i = 0; i < _box.Length; i++)
            {
                _box[i].X -= (int)xmin;
                _box[i].Y -= (int)ymin;
            }

            //cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

            Cv2.FillPoly(mask, new OpenCvSharp.Point[][] { _box.Select(p => new OpenCvSharp.Point((int)p.X, (int)p.Y)).ToArray() }, Scalar.White);

            Scalar mean = Cv2.Mean(bitmap[new Rect((int)xmin, (int)ymin, (int)(xmax - xmin + 1), (int)(ymax - ymin + 1))], mask);
            return mean.Val0;
        }
    }
}
