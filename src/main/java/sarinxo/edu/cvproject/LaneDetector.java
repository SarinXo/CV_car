package sarinxo.edu.cvproject;

import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

public class LaneDetector {

    public Mat detectLaneMarking(Mat frame) {

        Mat result = frame.clone();

        int height = frame.rows();
        int width = frame.cols();

        // 1. ROI (нижняя часть изображения)
        Rect roiRect = new Rect(0, height/2, width, height/2);
        Mat roi = new Mat(frame, roiRect);

        // 2. GRAY
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);

        // 3. Threshold (белая разметка)
        Mat mask1 = new Mat();
        Imgproc.threshold(gray, mask1, 200, 255, Imgproc.THRESH_BINARY);

        // 4. HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);

        Mat mask2 = new Mat();
        Core.inRange(
                hsv,
                new Scalar(0, 0, 180),
                new Scalar(255, 50, 255),
                mask2
        );

        // 5. Canny edges
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 50, 150);

        // 6. объединение масок
        Mat mask = new Mat();
        Core.bitwise_or(mask1, mask2, mask);
        Core.bitwise_or(mask, edges, mask);

        // 7. морфология (убрать шум)
        Mat kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_RECT,
                new Size(5,5)
        );

        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

        // 8. поиск контуров
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(
                mask,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        // 9. рисование контуров
        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            if (area > 300) { // фильтр шума

                List<MatOfPoint> single = new ArrayList<>();
                single.add(contour);

                Imgproc.drawContours(
                        roi,
                        single,
                        -1,
                        new Scalar(0,255,0),
                        -1 // закрашивание
                );
            }
        }

        roi.copyTo(result.submat(roiRect));

        return result;
    }

    public Mat detectLaneMarking2(Mat frame) {

        Mat result = frame.clone();

        int height = frame.rows();
        int width = frame.cols();

        Rect roiRect = new Rect(0, height/2, width, height/2);
        Mat roi = new Mat(frame, roiRect);

        // GRAY
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(gray, gray, new Size(5,5), 0);

        // Sobel X
        Mat sobelX = new Mat();
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0);

        Mat absSobelX = new Mat();
        Core.convertScaleAbs(sobelX, absSobelX);

        Mat sobelMask = new Mat();
        Imgproc.threshold(absSobelX, sobelMask, 30, 255, Imgproc.THRESH_BINARY); // БЫЛО 50

        // HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);

        // белая разметка
        Mat whiteMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(0, 0, 180),
                new Scalar(255, 60, 255),
                whiteMask
        );

        // желтая разметка
        Mat yellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 80, 80),
                new Scalar(40, 255, 255),
                yellowMask
        );

        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowMask, colorMask);

        // объединяем
        Mat mask = new Mat();
        Core.bitwise_and(colorMask, sobelMask, mask);

        // Canny
        Mat edges = new Mat();
        Imgproc.Canny(mask, edges, 40, 120);

        // Hough
        Mat lines = new Mat();

        Imgproc.HoughLinesP(
                edges,
                lines,
                1,
                Math.PI / 180,
                25,     // БЫЛО 40
                15,     // БЫЛО 40 (разрешает короткие линии)
                60      // БЫЛО 100
        );

        for (int i = 0; i < lines.rows(); i++) {

            double[] l = lines.get(i,0);

            int x1 = (int)l[0];
            int y1 = (int)l[1];
            int x2 = (int)l[2];
            int y2 = (int)l[3];

            double angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

            // расширенный диапазон
            if (Math.abs(angle) > 10 && Math.abs(angle) < 170) {

                Imgproc.line(
                        roi,
                        new Point(x1,y1),
                        new Point(x2,y2),
                        new Scalar(0,255,0),
                        3
                );
            }
        }

        roi.copyTo(result.submat(roiRect));

        return result;
    }

    public Mat detectLaneMarking3(Mat frame) {

        Mat result = frame.clone();

        int height = frame.rows();
        int width = frame.cols();

        Rect roiRect = new Rect(0, height/2, width, height/2);
        Mat roi = new Mat(frame, roiRect);

        // GRAY
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5,5), 0);

        // Sobel X
        Mat sobelX = new Mat();
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0);
        Mat absSobelX = new Mat();
        Core.convertScaleAbs(sobelX, absSobelX);

        Mat sobelMask = new Mat();
        Imgproc.threshold(absSobelX, sobelMask, 30, 255, Imgproc.THRESH_BINARY); // чуть чувствительнее

        // HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);

        // белая разметка
        Mat whiteMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(0, 0, 180),
                new Scalar(255, 60, 255),
                whiteMask
        );

        // желтая разметка
        Mat yellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 80, 80),
                new Scalar(40, 255, 255),
                yellowMask
        );

        // грязно-желтая / коричневая разметка
        Mat dirtyYellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(10, 30, 80),
                new Scalar(40, 200, 200),
                dirtyYellowMask
        );

        // объединяем все цветовые маски
        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowMask, colorMask);
        Core.bitwise_or(colorMask, dirtyYellowMask, colorMask);

        // объединяем цвет + Sobel
        Mat mask = new Mat();
        Core.bitwise_and(colorMask, sobelMask, mask);

        // Canny
        Mat edges = new Mat();
        Imgproc.Canny(mask, edges, 40, 120);

        // Hough
        Mat lines = new Mat();
        Imgproc.HoughLinesP(
                edges,
                lines,
                1,
                Math.PI / 180,
                25,
                5,
                60
        );

        // рисуем линии на ROI
        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i,0);

            int x1 = (int)l[0];
            int y1 = (int)l[1];
            int x2 = (int)l[2];
            int y2 = (int)l[3];

            double angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

            // расширенный диапазон
            if (Math.abs(angle) > 10 && Math.abs(angle) < 170) {
                Imgproc.line(
                        roi,
                        new Point(x1,y1),
                        new Point(x2,y2),
                        new Scalar(0,255,0),
                        3
                );
            }
        }

        roi.copyTo(result.submat(roiRect));
        return result;
    }

    public Mat detectLaneMarkingAdaptive(Mat frame) {

        Mat result = frame.clone();

        int height = frame.rows();
        int width = frame.cols();

        Rect roiRect = new Rect(0, height / 2, width, height / 2);
        Mat roi = new Mat(frame, roiRect);

        // --- Gray + Sobel X ---
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        Mat sobelX = new Mat();
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0);
        Mat absSobelX = new Mat();
        Core.convertScaleAbs(sobelX, absSobelX);

        Mat sobelMask = new Mat();
        Imgproc.threshold(absSobelX, sobelMask, 25, 255, Imgproc.THRESH_BINARY);

        // --- HSV ---
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);

        // белая разметка (сузили диапазон)
        Mat whiteMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(0, 0, 200),    // минимальная яркость увеличена
                new Scalar(180, 50, 255), // максимально допустимая насыщенность уменьшена
                whiteMask
        );

        // желтая разметка
        Mat yellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 80, 80),
                new Scalar(40, 255, 255),
                yellowMask
        );

        // грязно-желтая / коричневая разметка
        Mat dirtyYellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 30, 60),
                new Scalar(45, 200, 200),
                dirtyYellowMask
        );

        // объединяем желтую и грязно-желтую
        Mat yellowCombined = new Mat();
        Core.bitwise_or(yellowMask, dirtyYellowMask, yellowCombined);

        // LAB канал B
        Mat lab = new Mat();
        Imgproc.cvtColor(roi, lab, Imgproc.COLOR_BGR2Lab);
        List<Mat> labChannels = new ArrayList<>();
        Core.split(lab, labChannels);
        Mat bChannel = labChannels.get(2);
        Mat labMask = new Mat();
        Imgproc.threshold(bChannel, labMask, 140, 255, Imgproc.THRESH_BINARY);

        // объединяем все цветовые маски
        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowCombined, colorMask);
        Core.bitwise_or(colorMask, labMask, colorMask);

        // исключаем красный (фары)
        Mat redMask = new Mat();
        Core.inRange(hsv, new Scalar(0, 120, 150), new Scalar(10, 255, 255), redMask);
        Mat redMask2 = new Mat();
        Core.inRange(hsv, new Scalar(170, 120, 150), new Scalar(180, 255, 255), redMask2);
        Core.bitwise_or(redMask, redMask2, redMask);
        Core.bitwise_not(redMask, redMask);
        Core.bitwise_and(colorMask, redMask, colorMask);

        // объединяем цвет + Sobel
        Mat mask = new Mat();
        Core.bitwise_and(colorMask, sobelMask, mask);

        // Canny
        Mat edges = new Mat();
        Imgproc.Canny(mask, edges, 40, 120);

        // Hough
        Mat lines = new Mat();
        Imgproc.HoughLinesP(
                edges,
                lines,
                1,
                Math.PI / 180,
                25,
                15,
                60
        );

        // рисуем линии
        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);

            int x1 = (int) l[0];
            int y1 = (int) l[1];
            int x2 = (int) l[2];
            int y2 = (int) l[3];

            double angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

            if (Math.abs(angle) > 10 && Math.abs(angle) < 170) {
                Imgproc.line(
                        roi,
                        new Point(x1, y1),
                        new Point(x2, y2),
                        new Scalar(0, 255, 0),
                        3
                );
            }
        }

        roi.copyTo(result.submat(roiRect));
        return result;
    }

    public Mat detectLaneMarking4(Mat frame) {

        Mat result = frame.clone();

        int height = frame.rows();
        int width = frame.cols();

        Rect roiRect = new Rect(0, height / 2, width, height / 2);
        Mat roi = new Mat(frame, roiRect);

        // --- Gray + Sobel X ---
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        Mat sobelX = new Mat();
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0);
        Mat absSobelX = new Mat();
        Core.convertScaleAbs(sobelX, absSobelX);

        Mat sobelMask = new Mat();
        Imgproc.threshold(absSobelX, sobelMask, 25, 255, Imgproc.THRESH_BINARY); // чуть мягче

        // --- HSV ---
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);

        // Белая разметка (широкий диапазон)
        Mat whiteMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(0, 0, 160),
                new Scalar(180, 60, 255),
                whiteMask
        );

        // Желтая разметка (чистая)
        Mat yellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 80, 80),
                new Scalar(40, 255, 255),
                yellowMask
        );

        // Грязно-желтая / коричневая разметка (адаптивно)
        Mat dirtyYellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 30, 60), // низкая насыщенность и яркость
                new Scalar(45, 200, 200),
                dirtyYellowMask
        );

        // Объединяем желтую и грязно-желтую
        Mat yellowCombined = new Mat();
        Core.bitwise_or(yellowMask, dirtyYellowMask, yellowCombined);

        // --- LAB канал B для усиления желтого/коричневого ---
        Mat lab = new Mat();
        Imgproc.cvtColor(roi, lab, Imgproc.COLOR_BGR2Lab);
        List<Mat> labChannels = new ArrayList<>();
        Core.split(lab, labChannels);
        Mat bChannel = labChannels.get(2);
        Mat labMask = new Mat();
        Imgproc.threshold(bChannel, labMask, 140, 255, Imgproc.THRESH_BINARY);

        // --- Объединяем все цветовые маски ---
        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowCombined, colorMask);
        Core.bitwise_or(colorMask, labMask, colorMask);

        // --- Исключаем красный (фары) ---
        Mat redMask = new Mat();
        Core.inRange(hsv, new Scalar(0, 120, 150), new Scalar(10, 255, 255), redMask);
        Mat redMask2 = new Mat();
        Core.inRange(hsv, new Scalar(170, 120, 150), new Scalar(180, 255, 255), redMask2);
        Core.bitwise_or(redMask, redMask2, redMask);
        Core.bitwise_not(redMask, redMask);
        Core.bitwise_and(colorMask, redMask, colorMask);

        // --- объединяем цвет + Sobel ---
        Mat mask = new Mat();
        Core.bitwise_and(colorMask, sobelMask, mask);

        // --- Canny ---
        Mat edges = new Mat();
        Imgproc.Canny(mask, edges, 40, 120);

        // --- Hough ---
        Mat lines = new Mat();
        Imgproc.HoughLinesP(
                edges,
                lines,
                1,
                Math.PI / 180,
                25,
                15,
                60
        );

        // --- рисуем линии ---
        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);

            int x1 = (int) l[0];
            int y1 = (int) l[1];
            int x2 = (int) l[2];
            int y2 = (int) l[3];

            double angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

            if (Math.abs(angle) > 10 && Math.abs(angle) < 170) {
                Imgproc.line(
                        roi,
                        new Point(x1, y1),
                        new Point(x2, y2),
                        new Scalar(0, 255, 0),
                        3
                );
            }
        }

        roi.copyTo(result.submat(roiRect));
        return result;
    }

    public Mat detectLaneMarkingFinal(Mat frame) {

        Mat result = frame.clone();
        int height = frame.rows();
        int width = frame.cols();

        // ROI: нижняя половина
        Rect roiRect = new Rect(0, height / 2, width, height / 2);
        Mat roi = new Mat(frame, roiRect);

        // --- Gray + Sobel X ---
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        Mat sobelX = new Mat();
        Imgproc.Sobel(gray, sobelX, CvType.CV_16S, 1, 0);
        Mat absSobelX = new Mat();
        Core.convertScaleAbs(sobelX, absSobelX);

        Mat sobelMask = new Mat();
        Imgproc.threshold(absSobelX, sobelMask, 30, 255, Imgproc.THRESH_BINARY);

        // --- HSV ---
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);

        // Белая разметка (сузили диапазон)
        Mat whiteMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(0, 0, 200),
                new Scalar(180, 50, 255),
                whiteMask
        );

        // Желтая разметка
        Mat yellowMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 80, 80),
                new Scalar(40, 255, 255),
                yellowMask
        );

        // Грязно-желтая / коричневая разметка (до темно-желтого)
        Mat brownMask = new Mat();
        Core.inRange(
                hsv,
                new Scalar(15, 50, 80),
                new Scalar(45, 200, 200),
                brownMask
        );

        // объединяем желтую и коричневую
        Mat yellowCombined = new Mat();
        Core.bitwise_or(yellowMask, brownMask, yellowCombined);

        // LAB канал B для дополнительной фильтрации
        Mat lab = new Mat();
        Imgproc.cvtColor(roi, lab, Imgproc.COLOR_BGR2Lab);
        List<Mat> labChannels = new ArrayList<>();
        Core.split(lab, labChannels);
        Mat bChannel = labChannels.get(2);
        Mat labMask = new Mat();
        Imgproc.threshold(bChannel, labMask, 140, 255, Imgproc.THRESH_BINARY);

        // объединяем все цветовые маски
        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowCombined, colorMask);
        Core.bitwise_or(colorMask, labMask, colorMask);

        // исключаем красный (фары)
        Mat redMask1 = new Mat();
        Mat redMask2 = new Mat();
        Core.inRange(hsv, new Scalar(0, 120, 150), new Scalar(10, 255, 255), redMask1);
        Core.inRange(hsv, new Scalar(170, 120, 150), new Scalar(180, 255, 255), redMask2);
        Core.bitwise_or(redMask1, redMask2, redMask1);
        Core.bitwise_not(redMask1, redMask1);
        Core.bitwise_and(colorMask, redMask1, colorMask);

        // объединяем цвет + Sobel
        Mat mask = new Mat();
        Core.bitwise_and(colorMask, sobelMask, mask);

        // Canny
        Mat edges = new Mat();
        Imgproc.Canny(mask, edges, 50, 150);

        // HoughLinesP
        Mat lines = new Mat();
        Imgproc.HoughLinesP(
                edges,
                lines,
                1,
                Math.PI / 180,
                25,
                20,
                50
        );

        // разделяем линии на левую и правую по X
        List<Point[]> leftLines = new ArrayList<>();
        List<Point[]> rightLines = new ArrayList<>();
        int centerX = width / 2;

        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);
            int x1 = (int) l[0], y1 = (int) l[1], x2 = (int) l[2], y2 = (int) l[3];
            double angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

            if (Math.abs(angle) < 10 || Math.abs(angle) > 170) continue; // почти горизонтальные
            if (angle < 0) angle += 180; // нормализуем

            Point[] linePts = new Point[]{new Point(x1, y1), new Point(x2, y2)};

            if ((x1 + x2) / 2 < centerX) leftLines.add(linePts);
            else rightLines.add(linePts);
        }

        // Функция усреднения линии
        Function<List<Point[]>, Point[]> averageLine = (linesList) -> {
            if (linesList.isEmpty()) return null;

            double x1Sum = 0, y1Sum = 0, x2Sum = 0, y2Sum = 0;

            for (Point[] ln : linesList) {
                x1Sum += ln[0].x;
                y1Sum += ln[0].y;
                x2Sum += ln[1].x;
                y2Sum += ln[1].y;
            }

            int n = linesList.size();
            return new Point[]{new Point(x1Sum / n, y1Sum / n), new Point(x2Sum / n, y2Sum / n)};
        };

        Point[] leftAvg = averageLine.apply(leftLines);
        Point[] rightAvg = averageLine.apply(rightLines);

        // рисуем линии
        if (leftAvg != null)
            Imgproc.line(roi, leftAvg[0], leftAvg[1], new Scalar(0, 255, 0), 3);
        if (rightAvg != null)
            Imgproc.line(roi, rightAvg[0], rightAvg[1], new Scalar(0, 255, 0), 3);

        roi.copyTo(result.submat(roiRect));
        return result;
    }

    // --------------------------------
    // IMAGE ENHANCEMENT
    // --------------------------------

    private Mat enhance(Mat frame){

        Mat lab = new Mat();
        Imgproc.cvtColor(frame, lab, Imgproc.COLOR_BGR2Lab);

        List<Mat> channels = new ArrayList<>();
        Core.split(lab, channels);

        CLAHE clahe = Imgproc.createCLAHE();
        clahe.setClipLimit(3);

        Mat cl = new Mat();
        clahe.apply(channels.get(0), cl);

        channels.set(0, cl);

        Core.merge(channels, lab);

        Mat result = new Mat();
        Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2BGR);

        return result;
    }

    // --------------------------------
    // ROI
    // --------------------------------

    private Mat applyROI(Mat frame){

        Mat mask = Mat.zeros(frame.size(), CvType.CV_8UC1);

        int w = frame.width();
        int h = frame.height();

        Point p1 = new Point(w*0.1,h);
        Point p2 = new Point(w*0.9,h);
        Point p3 = new Point(w*0.6,h*0.6);
        Point p4 = new Point(w*0.4,h*0.6);

        MatOfPoint poly = new MatOfPoint(p1,p2,p3,p4);

        Imgproc.fillPoly(mask,Arrays.asList(poly),new Scalar(255));

        Mat result = new Mat();
        frame.copyTo(result,mask);

        return result;
    }

    // --------------------------------
    // WHITE LINE DETECTION
    // --------------------------------

    private Mat detectWhiteLines(Mat frame){

        Mat lab = new Mat();
        Imgproc.cvtColor(frame,lab,Imgproc.COLOR_BGR2Lab);

        List<Mat> ch = new ArrayList<>();
        Core.split(lab,ch);

        Mat L = ch.get(0);

        Mat white = new Mat();
        Imgproc.threshold(L,white,180,255,Imgproc.THRESH_BINARY);

        // удаляем крупные пятна (например отражения)
        removeLargeContours(white);

        return white;
    }

    // --------------------------------
    // EDGE DETECTION
    // --------------------------------

    private Mat detectEdges(Mat mask){

        Mat sobel = new Mat();

        Imgproc.Sobel(mask,sobel,CvType.CV_64F,1,0);

        Mat abs = new Mat();
        Core.convertScaleAbs(sobel,abs);

        Imgproc.threshold(abs,abs,30,255,Imgproc.THRESH_BINARY);

        return abs;
    }

    // --------------------------------
    // REMOVE REFLECTIONS
    // --------------------------------

    private void removeLargeContours(Mat mask){

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(mask,contours,hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE);

        for(MatOfPoint c:contours){

            Rect r = Imgproc.boundingRect(c);

            if(r.area() > 2000){

                Imgproc.drawContours(mask,
                        Arrays.asList(c),
                        -1,
                        new Scalar(0),
                        -1);
            }
        }
    }

    // --------------------------------
    // COLLECT EDGE POINTS
    // --------------------------------

    private List<Point> collectPoints(Mat edges){

        List<Point> pts = new ArrayList<>();

        for(int y=0;y<edges.rows();y++){

            for(int x=0;x<edges.cols();x++){

                if(edges.get(y,x)[0] > 0){

                    pts.add(new Point(x,y));
                }
            }
        }

        return pts;
    }

    // --------------------------------
    // FIT LINE
    // --------------------------------

    private Line fitLine(List<Point> pts,int width,boolean left){

        List<Point> filtered = new ArrayList<>();

        for(Point p:pts){

            if(left && p.x < width/2)
                filtered.add(p);

            if(!left && p.x > width/2)
                filtered.add(p);
        }

        if(filtered.size() < 50)
            return null;

        MatOfPoint2f mat = new MatOfPoint2f();
        mat.fromList(filtered);

        Mat line = new Mat();

        Imgproc.fitLine(mat,line,
                Imgproc.DIST_L2,
                0,
                0.01,
                0.01);

        double vx = line.get(0,0)[0];
        double vy = line.get(1,0)[0];
        double x = line.get(2,0)[0];
        double y = line.get(3,0)[0];

        Point p1 = new Point(x - vx*1000, y - vy*1000);
        Point p2 = new Point(x + vx*1000, y + vy*1000);

        return new Line(p1,p2);
    }

    // --------------------------------

    static class Line{

        Point p1;
        Point p2;

        Line(Point a,Point b){

            p1=a;
            p2=b;
        }
    }

}
