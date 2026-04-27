package sarinxo.edu.cvproject;
import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


public class LaneDetector2 {

    // Белый цвет
    private static final Scalar WHITE_LOW = new Scalar(0, 0, 190);
    private static final Scalar WHITE_HIGH = new Scalar(180, 65, 255);

    // Улучшенный диапазон грязно-жёлтого
    private static final Scalar YELLOW_LOW = new Scalar(10, 40, 60);
    private static final Scalar YELLOW_HIGH = new Scalar(45, 255, 255);

    // EMA сглаживание
    private static final double ALPHA = 0.22;

    private Lane smoothedLeftLane;
    private Lane smoothedRightLane;

    private static class Lane {
        double x1, y1, x2, y2;

        Lane(double x1, double y1, double x2, double y2) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
        }

        void update(Lane next) {
            x1 = ALPHA * next.x1 + (1.0 - ALPHA) * x1;
            y1 = ALPHA * next.y1 + (1.0 - ALPHA) * y1;
            x2 = ALPHA * next.x2 + (1.0 - ALPHA) * x2;
            y2 = ALPHA * next.y2 + (1.0 - ALPHA) * y2;
        }
    }

    public Mat detectLanes(Mat frame) {

        Mat result = frame.clone();

        int width = frame.cols();
        int height = frame.rows();

        // =====================================================
        // 1. CLAHE
        // =====================================================
        Mat enhanced = applyCLAHE(frame);

        // =====================================================
        // 2. HSV
        // =====================================================
        Mat hsv = new Mat();
        Imgproc.cvtColor(enhanced, hsv, Imgproc.COLOR_BGR2HSV);

        Mat whiteMask = new Mat();
        Mat yellowMask = new Mat();

        Core.inRange(hsv, WHITE_LOW, WHITE_HIGH, whiteMask);
        Core.inRange(hsv, YELLOW_LOW, YELLOW_HIGH, yellowMask);

        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowMask, colorMask);

        // =====================================================
        // 3. Морфология (сильнее фильтрация)
        // =====================================================
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                new Size(5, 5)
        );

        Imgproc.morphologyEx(
                colorMask,
                colorMask,
                Imgproc.MORPH_CLOSE,
                kernel
        );

        Imgproc.morphologyEx(
                colorMask,
                colorMask,
                Imgproc.MORPH_OPEN,
                kernel
        );

        // =====================================================
        // 4. ROI нижняя половина
        // =====================================================
        Rect roiRect = new Rect(0, height / 2, width, height / 2);
        Mat roi = new Mat(colorMask, roiRect);

        // =====================================================
        // 5. Blur + Canny (ниже чувствительность)
        // =====================================================
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(roi, blurred, new Size(7, 7), 0);

        Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, 70, 180);

        // =====================================================
        // 6. Hough (ниже чувствительность)
        // =====================================================
        Mat lines = new Mat();

        Imgproc.HoughLinesP(
                edges,
                lines,
                1,
                Math.PI / 180,
                35,     // threshold выше
                35,     // minLineLength выше
                20
        );

        Lane currentLeft = null;
        Lane currentRight = null;

        double bestLeftScore = 0;
        double bestRightScore = 0;

        for (int i = 0; i < lines.rows(); i++) {

            double[] l = lines.get(i, 0);

            double x1 = l[0];
            double y1 = l[1];
            double x2 = l[2];
            double y2 = l[3];

            double dx = x2 - x1;
            double dy = y2 - y1;

            if (Math.abs(dx) < 1.0) dx = 1.0;

            double slope = dy / dx;

            // меньше ложных срабатываний
            if (Math.abs(slope) < 0.55) continue;

            double length = Math.sqrt(dx * dx + dy * dy);
            double score = length * Math.abs(slope);

            double midX = (x1 + x2) * 0.5;

            if (midX < width * 0.5) {

                if (score > bestLeftScore) {
                    bestLeftScore = score;
                    currentLeft = new Lane(x1, y1, x2, y2);
                }

            } else {

                if (score > bestRightScore) {
                    bestRightScore = score;
                    currentRight = new Lane(x1, y1, x2, y2);
                }
            }
        }

        // =====================================================
        // 7. Защита от резких скачков
        // =====================================================
        double maxShift = 35.0;

        if (currentLeft != null && smoothedLeftLane != null) {

            if (distance(currentLeft, smoothedLeftLane) > maxShift) {
                currentLeft = smoothedLeftLane;
            }
        }

        if (currentRight != null && smoothedRightLane != null) {

            if (distance(currentRight, smoothedRightLane) > maxShift) {
                currentRight = smoothedRightLane;
            }
        }

        // =====================================================
        // 8. EMA сильнее сглаживание
        // =====================================================
        if (currentLeft != null) {
            if (smoothedLeftLane == null) smoothedLeftLane = currentLeft;
            else smoothedLeftLane.update(currentLeft);
        }

        if (currentRight != null) {
            if (smoothedRightLane == null) smoothedRightLane = currentRight;
            else smoothedRightLane.update(currentRight);
        }

        // =====================================================
        // 9. Draw
        // =====================================================
        if (smoothedLeftLane != null) {
            Imgproc.line(
                    result,
                    new Point(smoothedLeftLane.x1, smoothedLeftLane.y1 + height / 2),
                    new Point(smoothedLeftLane.x2, smoothedLeftLane.y2 + height / 2),
                    new Scalar(0, 255, 0),
                    4
            );
        }

        if (smoothedRightLane != null) {
            Imgproc.line(
                    result,
                    new Point(smoothedRightLane.x1, smoothedRightLane.y1 + height / 2),
                    new Point(smoothedRightLane.x2, smoothedRightLane.y2 + height / 2),
                    new Scalar(0, 0, 255),
                    4
            );
        }

        enhanced.release();
        hsv.release();
        whiteMask.release();
        yellowMask.release();
        colorMask.release();
        kernel.release();
        roi.release();
        blurred.release();
        edges.release();
        lines.release();

        return result;
    }

    private double distance(Lane a, Lane b) {
        return Math.sqrt(
                Math.pow(a.x1 - b.x1, 2) +
                        Math.pow(a.y1 - b.y1, 2) +
                        Math.pow(a.x2 - b.x2, 2) +
                        Math.pow(a.y2 - b.y2, 2)
        );
    }

    private Mat applyCLAHE(Mat src) {

        Mat lab = new Mat();
        Imgproc.cvtColor(src, lab, Imgproc.COLOR_BGR2Lab);

        List<Mat> channels = new ArrayList<>();
        Core.split(lab, channels);

        CLAHE clahe = Imgproc.createCLAHE(
                2.0,
                new Size(8, 8)
        );

        clahe.apply(channels.get(0), channels.get(0));

        Core.merge(channels, lab);

        Mat result = new Mat();
        Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2BGR);

        for (Mat c : channels) {
            c.release();
        }

        lab.release();

        return result;
    }
}
