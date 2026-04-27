package sarinxo.edu.cvproject;

import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.function.Function;

public class LaneDetector {
    // Константы цветов HSV
    private static final Scalar WHITE_LOW = new Scalar(0, 0, 200);
    private static final Scalar WHITE_HIGH = new Scalar(180, 50, 255);
    private static final Scalar YELLOW_LOW = new Scalar(15, 100, 100);
    private static final Scalar YELLOW_HIGH = new Scalar(35, 255, 255);

    // Коэффициент сглаживания (0.1 - очень плавно, 0.9 - почти без сглаживания)
    private static final double ALPHA = 0.2;

    // Храним состояние линий между кадрами
    private Lane smoothedLeftLane = null;
    private Lane smoothedRightLane = null;

    // Вспомогательный класс для хранения параметров линии
    private static class Lane {
        double x1, y1, x2, y2;

        Lane(double x1, double y1, double x2, double y2) {
            this.x1 = x1; this.y1 = y1; this.x2 = x2; this.y2 = y2;
        }

        // Метод для применения EMA сглаживания
        void update(double nx1, double ny1, double nx2, double ny2) {
            this.x1 = ALPHA * nx1 + (1 - ALPHA) * this.x1;
            this.y1 = ALPHA * ny1 + (1 - ALPHA) * this.y1;
            this.x2 = ALPHA * nx2 + (1 - ALPHA) * this.x2;
            this.y2 = ALPHA * ny2 + (1 - ALPHA) * this.y2;
        }
    }

    public Mat detectLanes(Mat frame) {
        Mat result = frame.clone();
        int height = frame.rows();
        int width = frame.cols();

        // --- 1. Препроцессинг (как в прошлом примере) ---
        Mat hsv = new Mat();
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

        Mat whiteMask = new Mat();
        Mat yellowMask = new Mat();
        Core.inRange(hsv, WHITE_LOW, WHITE_HIGH, whiteMask);
        Core.inRange(hsv, YELLOW_LOW, YELLOW_HIGH, yellowMask);

        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowMask, colorMask);

        // ROI: нижняя половина
        Mat roi = new Mat(colorMask, new Rect(0, height / 2, width, height / 2));

        Mat blurred = new Mat();
        Imgproc.GaussianBlur(roi, blurred, new Size(5, 5), 0);
        Mat edges = new Mat();
        Imgproc.Canny(blurred, edges, 50, 150);

        // --- 2. Поиск линий ---
        Mat lines = new Mat();
        Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, 50, 10);

        // Переменные для поиска "лучшей" линии в текущем кадре
        Lane currentLeft = null;
        Lane currentRight = null;
        double maxLeftLen = 0;
        double maxRightLen = 0;

        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);
            double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            double length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
            double slope = (x2 - x1 == 0) ? 999 : (y2 - y1) / (x2 - x1);

            if (Math.abs(slope) < 0.5) continue; // Игнорируем горизонтальные

            if ((x1 + x2) / 2 < width / 2) { // Левая сторона
                if (length > maxLeftLen) {
                    maxLeftLen = length;
                    currentLeft = new Lane(x1, y1, x2, y2);
                }
            } else { // Правая сторона
                if (length > maxRightLen) {
                    maxRightLen = length;
                    currentRight = new Lane(x1, y1, x2, y2);
                }
            }
        }

        // --- 3. Временная фильтрация (EMA) ---
        if (currentLeft != null) {
            if (smoothedLeftLane == null) smoothedLeftLane = currentLeft;
            else smoothedLeftLane.update(currentLeft.x1, currentLeft.y1, currentLeft.x2, currentLeft.y2);
        }

        if (currentRight != null) {
            if (smoothedRightLane == null) smoothedRightLane = currentRight;
            else smoothedRightLane.update(currentRight.x1, currentRight.y1, currentRight.x2, currentRight.y2);
        }

        // --- 4. Отрисовка сглаженных линий ---
        if (smoothedLeftLane != null) {
            Imgproc.line(result,
                    new Point(smoothedLeftLane.x1, smoothedLeftLane.y1 + height / 2),
                    new Point(smoothedLeftLane.x2, smoothedLeftLane.y2 + height / 2),
                    new Scalar(0, 255, 0), 3);
        }
        if (smoothedRightLane != null) {
            Imgproc.line(result,
                    new Point(smoothedRightLane.x1, smoothedRightLane.y1 + height / 2),
                    new Point(smoothedRightLane.x2, smoothedRightLane.y2 + height / 2),
                    new Scalar(0, 0, 255), 3);
        }

        // Очистка памяти
        hsv.release(); whiteMask.release(); yellowMask.release();
        colorMask.release(); roi.release(); blurred.release(); edges.release(); lines.release();

        return result;
    }
}
