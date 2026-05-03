package sarinxo.edu.cvproject.detection;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.*;

public class LaneDetector3 {
    private Mat M;     // Вид сверху
    private Mat Minv;  // исходное изображение

    private double[] prevLeftFit = null;
    private double[] prevRightFit = null;

    private int nwindows = 9; //Количество горизонтальных окон в sliding window
    private int margin = 100; //Ширина окна поиска вокруг текущей позиции линии
    private int minpix = 50; // Минимум пикселей, чтобы “переместить” окно

    public LaneDetector3(Point[] src, Point[] dst) {
        MatOfPoint2f srcMat = new MatOfPoint2f(src);
        MatOfPoint2f dstMat = new MatOfPoint2f(dst);
        M = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        Minv = Imgproc.getPerspectiveTransform(dstMat, srcMat);
    }

    public Mat processFrame(Mat frame) {

        // 1. ROI (нижняя половина)
        Rect roi = new Rect(0, frame.rows()/2, frame.cols(), frame.rows()/2);
        Mat cropped = new Mat(frame, roi);

        // 2. BEV
        Mat bev = new Mat();
        Imgproc.warpPerspective(cropped, bev, M, cropped.size());

        // 3. Binary mask
        Mat binary = getBinaryMask(bev);

        // 4. Найти линии
        List<Point> leftPts = new ArrayList<>();
        List<Point> rightPts = new ArrayList<>();

        findLanePixels(binary, leftPts, rightPts);

        // 5. Аппроксимация
        double[] leftFit = polyFit(leftPts);
        double[] rightFit = polyFit(rightPts);

        // fallback если плохо нашли
        if (leftFit == null && prevLeftFit != null) leftFit = prevLeftFit;
        if (rightFit == null && prevRightFit != null) rightFit = prevRightFit;

        // 6. сглаживание
        if (prevLeftFit != null && leftFit != null)
            leftFit = smooth(prevLeftFit, leftFit, 0.7);

        if (prevRightFit != null && rightFit != null)
            rightFit = smooth(prevRightFit, rightFit, 0.7);

        prevLeftFit = leftFit;
        prevRightFit = rightFit;

        // 7. отрисовка
        Mat laneMask = drawLane(bev.size(), leftFit, rightFit);

        // 8. обратно в оригинал
        Mat unwarp = new Mat();
        Imgproc.warpPerspective(laneMask, unwarp, Minv, cropped.size());

        Mat result = frame.clone();
        unwarp.copyTo(result.rowRange(frame.rows()/2, frame.rows()));

        return result;
    }

    // =========================
    // Binary mask
    // =========================
    private Mat getBinaryMask(Mat img) {

        Mat hls = new Mat();
        Imgproc.cvtColor(img, hls, Imgproc.COLOR_BGR2HLS);

        Mat white = new Mat();
        Mat yellow = new Mat();

        Core.inRange(hls, new Scalar(0,200,0), new Scalar(255,255,100), white);
        Core.inRange(hls, new Scalar(15,100,100), new Scalar(40,255,255), yellow);

        Mat gray = new Mat();
        Imgproc.cvtColor(img, gray, Imgproc.COLOR_BGR2GRAY);

        Mat sobel = new Mat();
        Imgproc.Sobel(gray, sobel, CvType.CV_64F, 1, 0);
        Core.convertScaleAbs(sobel, sobel);

        Mat colorMask = new Mat();
        Core.bitwise_or(white, yellow, colorMask);

        Mat binary = new Mat();
        Core.bitwise_or(colorMask, sobel, binary);

        // morphology
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5,5));
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);

        return binary;
    }

    // =========================
    // Sliding window
    // =========================
    private void findLanePixels(Mat binary, List<Point> leftPts, List<Point> rightPts) {

        int height = binary.rows();
        int width = binary.cols();

        Mat roi = binary.rowRange(height / 2, height);

        int[] hist = new int[width];

        for (int y = 0; y < roi.rows(); y++) {
            for (int x = 0; x < roi.cols(); x++) {

                double v = roi.get(y, x)[0];
                if (v > 0) {
                    hist[x]++;
                }
            }
        }

        int midpoint = width / 2;

        int leftxBase = argMax(hist, 0, midpoint);
        int rightxBase = argMax(hist, midpoint, width);

        int windowHeight = height / nwindows;

        int leftx = leftxBase;
        int rightx = rightxBase;

        // reset (IMPORTANT)
        leftPts.clear();
        rightPts.clear();

        // =========================
        // 3. Sliding windows
        // =========================
        for (int w = 0; w < nwindows; w++) {

            int winYLow = height - (w + 1) * windowHeight;
            int winYHigh = height - w * windowHeight;

            int winXLeftLow = leftx - margin;
            int winXLeftHigh = leftx + margin;

            int winXRightLow = rightx - margin;
            int winXRightHigh = rightx + margin;

            int leftCount = 0;
            int rightCount = 0;
            int leftSumX = 0;
            int rightSumX = 0;

            for (int y = winYLow; y < winYHigh; y++) {
                for (int x = 0; x < width; x++) {

                    double val = binary.get(y, x)[0];
                    if (val == 0) continue;

                    if (x > winXLeftLow && x < winXLeftHigh) {
                        leftPts.add(new Point(x, y));
                        leftCount++;
                        leftSumX += x;
                    }

                    if (x > winXRightLow && x < winXRightHigh) {
                        rightPts.add(new Point(x, y));
                        rightCount++;
                        rightSumX += x;
                    }
                }
            }

            // =========================
            // 4. Update window centers
            // =========================
            if (leftCount > minpix) {
                leftx = leftSumX / leftCount;
            }

            if (rightCount > minpix) {
                rightx = rightSumX / rightCount;
            }
        }
    }

    // =========================
    // Polyfit (2nd order)
    // =========================
    private double[] polyFit(List<Point> pts) {

        if (pts.size() < 50) return null;

        int n = pts.size();

        Mat A = new Mat(n, 3, CvType.CV_64F);
        Mat Y = new Mat(n, 1, CvType.CV_64F);

        for (int i = 0; i < n; i++) {
            double y = pts.get(i).y;
            double x = pts.get(i).x;

            A.put(i,0, y*y);
            A.put(i,1, y);
            A.put(i,2, 1);

            Y.put(i,0, x);
        }

        Mat At = new Mat();
        Core.transpose(A, At);

        Mat AtA = new Mat();
        Core.gemm(At, A, 1, new Mat(), 0, AtA);

        Mat AtY = new Mat();
        Core.gemm(At, Y, 1, new Mat(), 0, AtY);

        Mat coeffs = new Mat();
        Core.solve(AtA, AtY, coeffs);

        return new double[]{
                coeffs.get(0,0)[0],
                coeffs.get(1,0)[0],
                coeffs.get(2,0)[0]
        };
    }

    // =========================
    // Draw lane
    // =========================
    private Mat drawLane(Size size, double[] leftFit, double[] rightFit) {

        Mat out = Mat.zeros(size, CvType.CV_8UC3);

        if (leftFit == null || rightFit == null) return out;

        for (int y = 0; y < size.height; y++) {

            int lx = (int)(leftFit[0]*y*y + leftFit[1]*y + leftFit[2]);
            int rx = (int)(rightFit[0]*y*y + rightFit[1]*y + rightFit[2]);

            if (lx >= 0 && lx < size.width)
                Imgproc.circle(out, new Point(lx,y), 2, new Scalar(0,255,0), -1);

            if (rx >= 0 && rx < size.width)
                Imgproc.circle(out, new Point(rx,y), 2, new Scalar(0,255,0), -1);
        }

        return out;
    }

    // =========================
    // Utils
    // =========================
    private int argMax(int[] hist, int start, int end) {

        int maxIdx = start;
        int maxVal = -1;

        for (int i = start; i < end; i++) {
            if (hist[i] > maxVal) {
                maxVal = hist[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    private int meanX(List<Point> pts) {
        double sum = 0;
        for (Point p : pts) sum += p.x;
        return (int)(sum / pts.size());
    }

    private double[] smooth(double[] prev, double[] curr, double alpha) {
        double[] out = new double[3];
        for (int i = 0; i < 3; i++)
            out[i] = alpha * curr[i] + (1 - alpha) * prev[i];
        return out;
    }
}
