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
}
