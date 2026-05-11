package sarinxo.edu.cvproject.detection;

import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
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

    private double[] smooth(double[] prev, double[] curr, double alpha) {
        double[] out = new double[3];
        for (int i = 0; i < 3; i++)
            out[i] = alpha * curr[i] + (1 - alpha) * prev[i];
        return out;
    }

    public static Mat extractMarkings(Mat input) {

        // --- 1. Лёгкая нормализация (без агрессивного CLAHE) ---
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);

        // --- 2. Маска "белой разметки" через HSV ---
        // Настроено так, чтобы:
        // a1a0a1 (≈161,160,161) и 979798 (≈151,151,152) ПРОХОДИЛИ
        // но bbbab9 / d6d2d6 уходили дальше на отсев
        Mat whiteMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0, 0, 150),     // S низкая, V от среднего+
                new Scalar(180, 35, 255),
                whiteMask);

        // --- 3. Фильтр "настоящий белый" через близость каналов BGR ---
        // белая разметка: R ≈ G ≈ B
        List<Mat> bgr = new ArrayList<>();
        Core.split(input, bgr);

        Mat diff1 = new Mat();
        Mat diff2 = new Mat();
        Mat diff3 = new Mat();

        Core.absdiff(bgr.get(0), bgr.get(1), diff1);
        Core.absdiff(bgr.get(1), bgr.get(2), diff2);
        Core.absdiff(bgr.get(2), bgr.get(0), diff3);

        Mat colorDiff = new Mat();
        Core.add(diff1, diff2, colorDiff);
        Core.add(colorDiff, diff3, colorDiff);

        // маленькая разница → "серо-белый"
        Mat neutralMask = new Mat();
        Imgproc.threshold(colorDiff, neutralMask, 60, 255, Imgproc.THRESH_BINARY_INV);

        // --- 4. Убираем "слишком светлую дорогу" через локальный контраст ---
        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

        Mat gradX = new Mat();
        Mat gradY = new Mat();

        Imgproc.Sobel(gray, gradX, CvType.CV_16S, 1, 0);
        Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, 1);

        Mat absX = new Mat();
        Mat absY = new Mat();

        Core.convertScaleAbs(gradX, absX);
        Core.convertScaleAbs(gradY, absY);

        Mat gradient = new Mat();
        Core.addWeighted(absX, 0.5, absY, 0.5, 0, gradient);

        Mat edgeMask = new Mat();
        Imgproc.threshold(gradient, edgeMask, 15, 255, Imgproc.THRESH_BINARY);

        // --- 5. Комбинация ---
        Mat result = new Mat();

        // цвет + нейтральность (убирает цветной шум)
        Core.bitwise_and(whiteMask, neutralMask, result);

        // добавляем требование границы (убирает дорогу)
        Core.bitwise_and(result, edgeMask, result);

        // --- 6. ROI (чтобы не ловить небо) ---
        Mat roiMask = Mat.zeros(result.size(), CvType.CV_8UC1);
        Imgproc.rectangle(roiMask,
                new Point(0, result.rows() * 0.35),
                new Point(result.cols(), result.rows()),
                new Scalar(255),
                -1);

        Core.bitwise_and(result, roiMask, result);

        // --- 7. Лёгкая морфология ---
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(3, 3));

        Imgproc.morphologyEx(result, result,
                Imgproc.MORPH_CLOSE, kernel);

        return result;
    }

   /* public static Mat extractMarkings(Mat input) {

        // --- 1. Лёгкая нормализация (без агрессивного CLAHE) ---
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);

        // --- 2. Маска "белой разметки" через HSV ---
        // Настроено так, чтобы:
        // a1a0a1 (≈161,160,161) и 979798 (≈151,151,152) ПРОХОДИЛИ
        // но bbbab9 / d6d2d6 уходили дальше на отсев
        Mat whiteMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0, 0, 150),     // S низкая, V от среднего+
                new Scalar(180, 35, 255),
                whiteMask);

        // --- 3. Фильтр "настоящий белый" через близость каналов BGR ---
        // белая разметка: R ≈ G ≈ B
        List<Mat> bgr = new ArrayList<>();
        Core.split(input, bgr);

        Mat diffRG = new Mat();
        Mat diffGB = new Mat();
        Mat diffBR = new Mat();

        Core.absdiff(bgr.get(2), bgr.get(1), diffRG);
        Core.absdiff(bgr.get(1), bgr.get(0), diffGB);
        Core.absdiff(bgr.get(0), bgr.get(2), diffBR);

        Mat colorDiff = new Mat();
        Core.add(diffRG, diffGB, colorDiff);
        Core.add(colorDiff, diffBR, colorDiff);

        // маленькая разница → "серо-белый"
        Mat neutralMask = new Mat();
        Imgproc.threshold(colorDiff, neutralMask, 60, 255, Imgproc.THRESH_BINARY_INV);

        // --- 4. Убираем "слишком светлую дорогу" через локальный контраст ---
        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

        Mat gradX = new Mat();
        Mat gradY = new Mat();

        Imgproc.Sobel(gray, gradX, CvType.CV_16S, 1, 0);
        Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, 1);

        Mat absX = new Mat();
        Mat absY = new Mat();

        Core.convertScaleAbs(gradX, absX);
        Core.convertScaleAbs(gradY, absY);

        Mat gradient = new Mat();
        Core.addWeighted(absX, 0.5, absY, 0.5, 0, gradient);

        Mat edgeMask = new Mat();
        Imgproc.threshold(gradient, edgeMask, 15, 255, Imgproc.THRESH_BINARY);

        // --- 5. Комбинация ---
        Mat result = new Mat();

        // цвет + нейтральность (убирает цветной шум)
        Core.bitwise_and(whiteMask, neutralMask, result);

        // добавляем требование границы (убирает дорогу)
        Core.bitwise_and(result, edgeMask, result);

        // --- 6. ROI (чтобы не ловить небо) ---
        Mat roiMask = Mat.zeros(result.size(), CvType.CV_8UC1);
        Imgproc.rectangle(roiMask,
                new Point(0, result.rows() * 0.35),
                new Point(result.cols(), result.rows()),
                new Scalar(255),
                -1);

        Core.bitwise_and(result, roiMask, result);

        // --- 7. Лёгкая морфология ---
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(3, 3));

        Imgproc.morphologyEx(result, result,
                Imgproc.MORPH_CLOSE, kernel);

        return result;
    }

    public Mat extractMarkings(Mat input) {
        Mat lab = new Mat();
        Imgproc.cvtColor(input, lab, Imgproc.COLOR_BGR2Lab);

        java.util.List<Mat> labChannels = new java.util.ArrayList<>();
        Core.split(lab, labChannels);

        CLAHE clahe = Imgproc.createCLAHE(1.8, new Size(8, 8));
        clahe.apply(labChannels.get(0), labChannels.get(0));

        Core.merge(labChannels, lab);

        Mat enhanced = new Mat();
        Imgproc.cvtColor(lab, enhanced, Imgproc.COLOR_Lab2BGR);

        // === 2. HSV ===
        Mat hsv = new Mat();
        Imgproc.cvtColor(enhanced, hsv, Imgproc.COLOR_BGR2HSV);

        // === 3. Цветовые маски (ослаблены) ===
        Mat pureWhite = new Mat();
        Core.inRange(hsv,
                new Scalar(0, 0, 150),
                new Scalar(180, 35, 255),
                pureWhite);

        Mat yellowishWhite = new Mat();
        Core.inRange(hsv,
                new Scalar(12, 20, 160),
                new Scalar(45, 140, 255),
                yellowishWhite);

        Core.subtract(yellowishWhite, pureWhite, yellowishWhite);

        Mat colorMask = new Mat();
        Core.bitwise_or(pureWhite, yellowishWhite, colorMask);

        // === 4. МЯГКИЙ градиент ===
        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        Mat gradX = new Mat();
        Mat gradY = new Mat();

        Imgproc.Sobel(gray, gradX, CvType.CV_16S, 1, 0);
        Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, 1);

        Mat absX = new Mat();
        Mat absY = new Mat();

        Core.convertScaleAbs(gradX, absX);
        Core.convertScaleAbs(gradY, absY);

        Mat gradient = new Mat();
        Core.addWeighted(absX, 0.5, absY, 0.5, 0, gradient);

        // ↓ ПОРОГ СИЛЬНО СНИЖЕН
        Mat edgeMask = new Mat();
        Imgproc.threshold(gradient, edgeMask, 20, 255, Imgproc.THRESH_BINARY);

        // ↓ НЕ ЖЕСТКОЕ AND, а "усиление"
        Mat result = new Mat();
        Core.bitwise_and(colorMask, edgeMask, result);
        Core.bitwise_or(result, colorMask, result);

        // === 5. ROI мягче (оставляем больше) ===
        Mat roiMask = Mat.zeros(result.size(), CvType.CV_8UC1);
        Imgproc.rectangle(roiMask,
                new Point(0, result.rows() * 0.3),
                new Point(result.cols(), result.rows()),
                new Scalar(255),
                -1);

        Core.bitwise_and(result, roiMask, result);

        // === 6. Морфология мягче ===
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(5, 5));

        Imgproc.morphologyEx(result, result,
                Imgproc.MORPH_CLOSE, kernel);

        return result;
    }*/
}
