package sarinxo.edu.cvproject.detection;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.*;

public class PerspectiveInitializer {

    @Data
    @AllArgsConstructor
    public static class Result {
        public Point[] src;
        public Point[] dst;
    }

    public Result initialize(Mat frame) {

        int h = frame.rows();
        int w = frame.cols();

        // 1. ROI (нижняя часть)
        Rect roiRect = new Rect(0, (int)(h * 0.6), w, (int)(h * 0.4));
        Mat roi = new Mat(frame, roiRect);

        // 2. grayscale + blur
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5,5), 0);

        // 3. edges
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 50, 150);

        // 4. histogram по X
        Mat hist = new Mat();
        Core.reduce(edges, hist, 0, Core.REDUCE_SUM, CvType.CV_32S);

        int center = w / 2;

        int leftX = argMax(hist, 0, center);
        int rightX = argMax(hist, center, w);

        // fallback если не нашли
        if (leftX <= 0) leftX = (int)(w * 0.3);
        if (rightX <= 0) rightX = (int)(w * 0.7);

        // перевод координат из ROI в полный кадр
        int bottomY = roiRect.y + roiRect.height;
        int topY = (int)(h * 0.6);

        // 5. строим src (трапеция)
        int offsetTop = 80;     // сужение сверху
        int offsetBottom = 20;  // расширение снизу

        Point bottomLeft  = new Point(leftX - offsetBottom, bottomY);
        Point bottomRight = new Point(rightX + offsetBottom, bottomY);

        Point topLeft  = new Point(leftX + offsetTop, topY);
        Point topRight = new Point(rightX - offsetTop, topY);

        // защита от выхода за границы
        bottomLeft.x  = clamp(bottomLeft.x, 0, w);
        bottomRight.x = clamp(bottomRight.x, 0, w);
        topLeft.x     = clamp(topLeft.x, 0, w);
        topRight.x    = clamp(topRight.x, 0, w);

        Point[] src = new Point[]{
                topLeft,
                topRight,
                bottomRight,
                bottomLeft
        };

        // 6. dst (прямоугольник)
        int margin = (int)(w * 0.25);

        Point[] dst = new Point[]{
                new Point(margin, 0),
                new Point(w - margin, 0),
                new Point(w - margin, h),
                new Point(margin, h)
        };

        return new Result(src, dst);
    }

    // =========================
    // Utils
    // =========================

    private int argMax(Mat hist, int start, int end) {
        int maxIdx = start;
        double maxVal = -1;

        for (int i = start; i < end; i++) {
            double v = hist.get(0, i)[0];
            if (v > maxVal) {
                maxVal = v;
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private double clamp(double val, double min, double max) {
        return Math.max(min, Math.min(max, val));
    }











    public static Mat extractMarkings(Mat input) {

        // === 1. HSV ===
        Mat hsv = new Mat();
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);

        // --- белый (расширенный) ---
        Mat whiteMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0, 0, 150),
                new Scalar(180, 50, 255),
                whiteMask);

        // === 2. "нейтральный цвет" (R≈G≈B) ===
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

        Mat neutralMask = new Mat();
        Imgproc.threshold(colorDiff, neutralMask, 80, 255, Imgproc.THRESH_BINARY_INV);

        // === 3. ЛОКАЛЬНЫЙ КОНТРАСТ (КЛЮЧЕВОЙ ФИКС) ===
        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

        // среднее
        Mat mean = new Mat();
        Imgproc.blur(gray, mean, new Size(9, 9));

        // mean(gray^2)
        Mat graySq = new Mat();
        Core.multiply(gray, gray, graySq);

        Mat meanSq = new Mat();
        Imgproc.blur(graySq, meanSq, new Size(9, 9));

        // variance = mean(x^2) - mean(x)^2
        Mat meanPow2 = new Mat();
        Core.multiply(mean, mean, meanPow2);

        Mat variance = new Mat();
        Core.subtract(meanSq, meanPow2, variance);

        // нормализация (для стабильного порога)
        Core.normalize(variance, variance, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);

        // маска "есть текстура"
        Mat textureMask = new Mat();
        Imgproc.threshold(variance, textureMask, 20, 255, Imgproc.THRESH_BINARY);

        // === 4. Комбинация ===
        Mat result = new Mat();

        Core.bitwise_and(whiteMask, neutralMask, result);

        // КЛЮЧ: убирает "гладкий светлый асфальт"
        Core.bitwise_and(result, textureMask, result);

        // === 5. ROI ===
        Mat roiMask = Mat.zeros(result.size(), CvType.CV_8UC1);
        Imgproc.rectangle(roiMask,
                new Point(0, result.rows() * 0.35),
                new Point(result.cols(), result.rows()),
                new Scalar(255),
                -1);

        Core.bitwise_and(result, roiMask, result);

        // === 6. Морфология ===
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT, new Size(3, 3));

        Imgproc.morphologyEx(result, result,
                Imgproc.MORPH_CLOSE, kernel);

        return result;
    }

}
