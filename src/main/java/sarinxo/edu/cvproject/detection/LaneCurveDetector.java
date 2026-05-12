package sarinxo.edu.cvproject.detection;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Detects left and right lane curves by sliding a window over a bird's-eye-view (BEV)
 * projection of the binary lane-marking mask.
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Apply a trapezoidal ROI to the mask.</li>
 *   <li>Warp the ROI rectangle to a top-down BEV — lanes become approximately vertical and
 *       parallel, so a quadratic in BEV space describes any curve / roundabout faithfully.</li>
 *   <li>Sum the bottom half of BEV column-wise into a 1D histogram. The two strongest peaks
 *       (separated by at least {@link Config#minLaneSeparationRatio}) give the starting x of
 *       the left and right lane lines. This works for any camera offset because no fixed
 *       centre is assumed.</li>
 *   <li>For each base, slide a fixed-width window upward through BEV. In each step gather
 *       non-zero pixels and recentre the window on their mean x if the cluster is dense
 *       enough — this lets the search follow strong curves.</li>
 *   <li>Fit a polynomial x = a·y² + b·y + c to each side's collected pixels (BEV space).</li>
 *   <li>Sample the polynomial along y, warp the samples back to image space, draw a polyline.</li>
 * </ol>
 *
 * <p>Not thread-safe; one instance per processing thread.
 */
public final class LaneCurveDetector {

    // =========================================================================================
    // Configuration
    // =========================================================================================

    public static final class Config {

        // -- ROI trapezoid (fractions of frame size) --
        public final double roiBottomY;
        public final double roiTopY;
        public final double roiBottomHalfWidth;
        public final double roiTopHalfWidth;

        // -- BEV target size --
        public final int bevWidth;
        public final int bevHeight;

        // -- Histogram base detection --
        public final int    histogramMinPeakSum;       // min column-sum to accept as a base
        public final double minLaneSeparationRatio;    // min |xL − xR| / bevWidth

        // -- Sliding window --
        public final int slidingWindowCount;
        public final int slidingWindowMarginX;         // half-width in BEV pixels
        public final int slidingWindowMinPixels;       // min pixels in window to recentre

        // -- Polynomial --
        public final int polynomialDegree;
        public final int minPointsForFit;

        // -- Drawing --
        public final Scalar  leftColor;
        public final Scalar  rightColor;
        public final int     lineThickness;
        public final int     drawStepPixels;
        public final boolean drawFilledArea;
        public final Scalar  filledAreaColor;
        public final double  filledAreaOpacity;

        private Config(Builder b) {
            this.roiBottomY              = b.roiBottomY;
            this.roiTopY                 = b.roiTopY;
            this.roiBottomHalfWidth      = b.roiBottomHalfWidth;
            this.roiTopHalfWidth         = b.roiTopHalfWidth;
            this.bevWidth                = b.bevWidth;
            this.bevHeight               = b.bevHeight;
            this.histogramMinPeakSum     = b.histogramMinPeakSum;
            this.minLaneSeparationRatio  = b.minLaneSeparationRatio;
            this.slidingWindowCount      = b.slidingWindowCount;
            this.slidingWindowMarginX    = b.slidingWindowMarginX;
            this.slidingWindowMinPixels  = b.slidingWindowMinPixels;
            this.polynomialDegree        = b.polynomialDegree;
            this.minPointsForFit         = b.minPointsForFit;
            this.leftColor               = b.leftColor;
            this.rightColor              = b.rightColor;
            this.lineThickness           = b.lineThickness;
            this.drawStepPixels          = b.drawStepPixels;
            this.drawFilledArea          = b.drawFilledArea;
            this.filledAreaColor         = b.filledAreaColor;
            this.filledAreaOpacity       = b.filledAreaOpacity;
        }

        public static Config  defaults() { return new Builder().build(); }
        public static Builder builder()  { return new Builder(); }

        public static final class Builder {
            // ROI — wide bottom captures lanes regardless of camera horizontal offset;
            // wider top tolerates moderately curved or off-centre roads.
            private double  roiBottomY              = 0.95;
            private double  roiTopY                 = 0.62;
            private double  roiBottomHalfWidth      = 0.48;
            private double  roiTopHalfWidth         = 0.18;

            // BEV resolution — independent of input frame size.
            private int     bevWidth                = 400;
            private int     bevHeight               = 600;

            // Histogram peak threshold. Mask pixels are 0/255, so a column-sum value of
            // (N × 255) means N white pixels accumulated in that column of the BEV bottom half.
            // 25 pixels is enough to dismiss isolated noise but accept short dashes.
            private int     histogramMinPeakSum     = 25 * 255;
            private double  minLaneSeparationRatio  = 0.20;

            // Sliding window — 9 vertical slices is the standard Udacity-style setup.
            private int     slidingWindowCount      = 9;
            private int     slidingWindowMarginX    = 60;
            private int     slidingWindowMinPixels  = 40;

            // Polynomial — quadratic captures any single-radius curve, including roundabouts
            // within the visible ROI.
            private int     polynomialDegree        = 2;
            private int     minPointsForFit         = 20;

            // Drawing — BGR. Yellow left, red right.
            private Scalar  leftColor               = new Scalar(0,   255, 255);
            private Scalar  rightColor              = new Scalar(0,   0,   255);
            private int     lineThickness           = 6;
            private int     drawStepPixels          = 6;
            private boolean drawFilledArea          = false;
            private Scalar  filledAreaColor         = new Scalar(0,   200, 0);
            private double  filledAreaOpacity       = 0.25;

            public Builder roiBottomY(double v)              { this.roiBottomY = v; return this; }
            public Builder roiTopY(double v)                 { this.roiTopY = v; return this; }
            public Builder roiBottomHalfWidth(double v)      { this.roiBottomHalfWidth = v; return this; }
            public Builder roiTopHalfWidth(double v)         { this.roiTopHalfWidth = v; return this; }
            public Builder bevWidth(int v)                   { this.bevWidth = v; return this; }
            public Builder bevHeight(int v)                  { this.bevHeight = v; return this; }
            public Builder histogramMinPeakSum(int v)        { this.histogramMinPeakSum = v; return this; }
            public Builder minLaneSeparationRatio(double v)  { this.minLaneSeparationRatio = v; return this; }
            public Builder slidingWindowCount(int v)         { this.slidingWindowCount = v; return this; }
            public Builder slidingWindowMarginX(int v)       { this.slidingWindowMarginX = v; return this; }
            public Builder slidingWindowMinPixels(int v)     { this.slidingWindowMinPixels = v; return this; }
            public Builder polynomialDegree(int v)           { this.polynomialDegree = v; return this; }
            public Builder minPointsForFit(int v)            { this.minPointsForFit = v; return this; }
            public Builder leftColor(Scalar v)               { this.leftColor = v; return this; }
            public Builder rightColor(Scalar v)              { this.rightColor = v; return this; }
            public Builder lineThickness(int v)              { this.lineThickness = v; return this; }
            public Builder drawStepPixels(int v)             { this.drawStepPixels = v; return this; }
            public Builder drawFilledArea(boolean v)         { this.drawFilledArea = v; return this; }
            public Builder filledAreaColor(Scalar v)         { this.filledAreaColor = v; return this; }
            public Builder filledAreaOpacity(double v)       { this.filledAreaOpacity = v; return this; }

            public Config build() {
                if (roiBottomY <= roiTopY)
                    throw new IllegalArgumentException("roiBottomY must be > roiTopY");
                if (bevWidth < 50 || bevHeight < 50)
                    throw new IllegalArgumentException("BEV dimensions too small");
                if (slidingWindowCount < 3)
                    throw new IllegalArgumentException("slidingWindowCount must be >= 3");
                if (slidingWindowMarginX <= 0)
                    throw new IllegalArgumentException("slidingWindowMarginX must be > 0");
                if (polynomialDegree != 1 && polynomialDegree != 2)
                    throw new IllegalArgumentException("polynomialDegree must be 1 or 2");
                if (minPointsForFit < polynomialDegree + 1)
                    throw new IllegalArgumentException("minPointsForFit must be > polynomialDegree");
                if (minLaneSeparationRatio < 0 || minLaneSeparationRatio > 1)
                    throw new IllegalArgumentException("minLaneSeparationRatio must be in [0,1]");
                if (filledAreaOpacity < 0 || filledAreaOpacity > 1)
                    throw new IllegalArgumentException("filledAreaOpacity must be in [0,1]");
                Objects.requireNonNull(leftColor);
                Objects.requireNonNull(rightColor);
                return new Config(this);
            }
        }
    }

    // =========================================================================================
    // State
    // =========================================================================================

    private final Config config;

    private int     lastWidth        = -1;
    private int     lastHeight       = -1;
    private Point[] roiPolygonImage  = null;
    private Mat     origToBevMatrix  = null;
    private Mat     bevToOrigMatrix  = null;

    private boolean released = false;

    // -- Diagnostics --
    public static final class Diagnostics {
        public final int     leftBaseX;
        public final int     rightBaseX;
        public final int     leftPixels;
        public final int     rightPixels;
        public final boolean leftDetected;
        public final boolean rightDetected;

        Diagnostics(int lb, int rb, int lp, int rp, boolean ld, boolean rd) {
            this.leftBaseX     = lb;
            this.rightBaseX    = rb;
            this.leftPixels    = lp;
            this.rightPixels   = rp;
            this.leftDetected  = ld;
            this.rightDetected = rd;
        }

        @Override public String toString() {
            return String.format(
                    "Diagnostics{leftBaseX=%d, rightBaseX=%d, leftPixels=%d, rightPixels=%d, "
                    + "leftDetected=%s, rightDetected=%s}",
                    leftBaseX, rightBaseX, leftPixels, rightPixels, leftDetected, rightDetected);
        }
    }
    private Diagnostics lastDiagnostics = null;
    public Diagnostics getLastDiagnostics() { return lastDiagnostics; }

    // =========================================================================================
    // Construction
    // =========================================================================================

    public LaneCurveDetector()              { this(Config.defaults()); }
    public LaneCurveDetector(Config config) { this.config = Objects.requireNonNull(config); }

    // =========================================================================================
    // Public API
    // =========================================================================================

    public Mat process(Mat frame, Mat mask) {
        if (released) throw new IllegalStateException("Detector has been released");
        Objects.requireNonNull(frame, "frame");
        Objects.requireNonNull(mask,  "mask");
        if (frame.empty() || mask.empty())
            throw new IllegalArgumentException("Inputs must not be empty");
        if (frame.channels() != 3)
            throw new IllegalArgumentException("frame must be CV_8UC3");
        if (mask.channels() != 1)
            throw new IllegalArgumentException("mask must be CV_8UC1");
        if (frame.width() != mask.width() || frame.height() != mask.height())
            throw new IllegalArgumentException("frame and mask must have identical dimensions");

        ensureGeometry(frame.width(), frame.height());

        Mat roiMask = applyRoi(mask);
        Mat bev     = new Mat();
        try {
            Imgproc.warpPerspective(roiMask, bev, origToBevMatrix,
                    new Size(config.bevWidth, config.bevHeight), Imgproc.INTER_NEAREST);

            int[] bases = findBases(bev);
            int leftBaseX  = bases[0];
            int rightBaseX = bases[1];

            List<Point> leftPixels  = (leftBaseX  >= 0)
                    ? slidingWindowSearch(bev, leftBaseX)  : new ArrayList<>();
            List<Point> rightPixels = (rightBaseX >= 0)
                    ? slidingWindowSearch(bev, rightBaseX) : new ArrayList<>();

            double[] leftCoef  = (leftPixels.size()  >= config.minPointsForFit)
                    ? fitPolynomial(leftPixels,  config.polynomialDegree)  : null;
            double[] rightCoef = (rightPixels.size() >= config.minPointsForFit)
                    ? fitPolynomial(rightPixels, config.polynomialDegree) : null;

            Mat output = frame.clone();
            drawLanes(output, leftCoef, rightCoef, leftPixels, rightPixels);

            lastDiagnostics = new Diagnostics(
                    leftBaseX, rightBaseX,
                    leftPixels.size(), rightPixels.size(),
                    leftCoef != null, rightCoef != null);

            return output;
        } finally {
            bev.release();
            roiMask.release();
        }
    }

    public void reset() {
        lastWidth  = -1;
        lastHeight = -1;
        releaseCachedGeometry();
    }

    public void release() {
        if (released) return;
        releaseCachedGeometry();
        released = true;
    }

    private void releaseCachedGeometry() {
        if (origToBevMatrix != null) { origToBevMatrix.release(); origToBevMatrix = null; }
        if (bevToOrigMatrix != null) { bevToOrigMatrix.release(); bevToOrigMatrix = null; }
        roiPolygonImage = null;
    }

    // =========================================================================================
    // Geometry / ROI
    // =========================================================================================

    private void ensureGeometry(int w, int h) {
        if (w == lastWidth && h == lastHeight) return;
        releaseCachedGeometry();

        double cx          = w * 0.5;
        double bottomY     = h * config.roiBottomY;
        double topY        = h * config.roiTopY;
        double bottomHalf  = w * config.roiBottomHalfWidth;
        double topHalf     = w * config.roiTopHalfWidth;

        roiPolygonImage = new Point[] {
                new Point(cx - bottomHalf, bottomY),
                new Point(cx - topHalf,    topY),
                new Point(cx + topHalf,    topY),
                new Point(cx + bottomHalf, bottomY)
        };

        Point[] dst = new Point[] {
                new Point(0,                  config.bevHeight - 1),
                new Point(0,                  0),
                new Point(config.bevWidth - 1, 0),
                new Point(config.bevWidth - 1, config.bevHeight - 1)
        };
        MatOfPoint2f src2f = new MatOfPoint2f(roiPolygonImage);
        MatOfPoint2f dst2f = new MatOfPoint2f(dst);
        try {
            origToBevMatrix = Imgproc.getPerspectiveTransform(src2f, dst2f);
            bevToOrigMatrix = Imgproc.getPerspectiveTransform(dst2f, src2f);
        } finally {
            src2f.release();
            dst2f.release();
        }

        lastWidth  = w;
        lastHeight = h;
    }

    private Mat applyRoi(Mat mask) {
        Mat roi = Mat.zeros(mask.size(), mask.type());
        MatOfPoint poly = new MatOfPoint(roiPolygonImage);
        try {
            Imgproc.fillPoly(roi, Collections.singletonList(poly), new Scalar(255));
            Mat result = new Mat();
            Core.bitwise_and(mask, roi, result);
            return result;
        } finally {
            roi.release();
            poly.release();
        }
    }

    // =========================================================================================
    // Base detection — find left and right lane starting x by histogram peaks
    // =========================================================================================

    /**
     * Returns {@code {leftBaseX, rightBaseX}} in BEV coordinates. Either value may be {@code -1}
     * when that side cannot be located. The two strongest peaks of the bottom-half column-sum
     * histogram are picked, with a non-maximum-suppression band of
     * {@code bevWidth × minLaneSeparationRatio} between them.
     */
    private int[] findBases(Mat bev) {
        Mat bottomHalf = bev.submat(bev.rows() / 2, bev.rows(), 0, bev.cols());
        Mat hMat       = new Mat();
        try {
            Core.reduce(bottomHalf, hMat, 0, Core.REDUCE_SUM, CvType.CV_32S);
            int w = hMat.cols();
            int[] hist = new int[w];
            hMat.get(0, 0, hist);

            int minSep = (int) (w * config.minLaneSeparationRatio);
            int minVal = config.histogramMinPeakSum;

            int max1Val = -1, max1X = -1;
            for (int x = 0; x < w; x++) {
                if (hist[x] > max1Val) { max1Val = hist[x]; max1X = x; }
            }
            if (max1Val < minVal) return new int[]{-1, -1};

            int max2Val = -1, max2X = -1;
            for (int x = 0; x < w; x++) {
                if (Math.abs(x - max1X) < minSep) continue;
                if (hist[x] > max2Val) { max2Val = hist[x]; max2X = x; }
            }

            if (max2Val < minVal) {
                // Only one peak — assign by position relative to BEV centre.
                boolean isLeft = max1X < w / 2;
                return new int[]{ isLeft ? max1X : -1, isLeft ? -1 : max1X };
            }

            return (max1X < max2X) ? new int[]{max1X, max2X} : new int[]{max2X, max1X};
        } finally {
            bottomHalf.release();
            hMat.release();
        }
    }

    // =========================================================================================
    // Sliding window — collect lane pixels by walking upward from the base
    // =========================================================================================

    private List<Point> slidingWindowSearch(Mat bev, int baseX) {
        int H      = bev.rows();
        int W      = bev.cols();
        int nWin   = config.slidingWindowCount;
        int winH   = H / nWin;
        int margin = config.slidingWindowMarginX;
        int minPix = config.slidingWindowMinPixels;

        int currentX = baseX;
        List<Point> collected = new ArrayList<>();
        MatOfPoint nonZero = new MatOfPoint();
        try {
            for (int wIdx = 0; wIdx < nWin; wIdx++) {
                int yLow  = Math.max(0, H - (wIdx + 1) * winH);
                int yHigh = H - wIdx * winH;
                int xLow  = Math.max(0, currentX - margin);
                int xHigh = Math.min(W, currentX + margin);
                if (yLow >= yHigh || xLow >= xHigh) continue;

                Mat slice = bev.submat(yLow, yHigh, xLow, xHigh);
                try {
                    Core.findNonZero(slice, nonZero);
                    int n = (int) nonZero.total();
                    if (n == 0) continue;
                    Point[] pts = nonZero.toArray();
                    long sumX = 0;
                    for (Point p : pts) {
                        int absX = xLow + (int) p.x;
                        int absY = yLow + (int) p.y;
                        collected.add(new Point(absX, absY));
                        sumX += absX;
                    }
                    if (n >= minPix) currentX = (int) (sumX / n);
                } finally {
                    slice.release();
                }
            }
        } finally {
            nonZero.release();
        }
        return collected;
    }

    // =========================================================================================
    // Drawing — sample polynomial in BEV, warp back to image, draw polylines
    // =========================================================================================

    private void drawLanes(Mat output, double[] leftCoef, double[] rightCoef,
                           List<Point> leftPixels, List<Point> rightPixels) {
        Point[] leftImg  = (leftCoef  != null) ? bevCurveToImage(leftCoef,  leftPixels)  : null;
        Point[] rightImg = (rightCoef != null) ? bevCurveToImage(rightCoef, rightPixels) : null;

        if (config.drawFilledArea && leftImg != null && rightImg != null) {
            drawFilledArea(output, leftImg, rightImg);
        }
        if (leftImg  != null) drawPolyline(output, leftImg,  config.leftColor);
        if (rightImg != null) drawPolyline(output, rightImg, config.rightColor);
    }

    /**
     * Samples the BEV-space polynomial from the topmost collected pixel down to the bottom of
     * the BEV, then warps the sample points back to image space.
     */
    private Point[] bevCurveToImage(double[] coef, List<Point> bevPoints) {
        if (bevPoints.isEmpty()) return null;
        int yMin = Integer.MAX_VALUE;
        for (Point p : bevPoints) if (p.y < yMin) yMin = (int) p.y;
        int yMax = config.bevHeight - 1;

        int step = Math.max(1, config.drawStepPixels);
        List<Point> samples = new ArrayList<>();
        for (int y = yMin; y <= yMax; y += step) {
            samples.add(new Point(evalPoly(coef, y), y));
        }
        if (samples.size() < 2) return null;

        MatOfPoint2f src2f = new MatOfPoint2f(samples.toArray(new Point[0]));
        MatOfPoint2f dst2f = new MatOfPoint2f();
        try {
            Core.perspectiveTransform(src2f, dst2f, bevToOrigMatrix);
            return dst2f.toArray();
        } finally {
            src2f.release();
            dst2f.release();
        }
    }

    private void drawPolyline(Mat output, Point[] pts, Scalar color) {
        MatOfPoint mop = new MatOfPoint(pts);
        try {
            Imgproc.polylines(output, Collections.singletonList(mop), false,
                    color, config.lineThickness, Imgproc.LINE_AA, 0);
        } finally {
            mop.release();
        }
    }

    private void drawFilledArea(Mat output, Point[] leftPts, Point[] rightPts) {
        Point[] poly = new Point[leftPts.length + rightPts.length];
        System.arraycopy(leftPts, 0, poly, 0, leftPts.length);
        for (int i = 0; i < rightPts.length; i++) {
            poly[leftPts.length + i] = rightPts[rightPts.length - 1 - i];
        }
        MatOfPoint mop = new MatOfPoint(poly);
        Mat overlay = output.clone();
        try {
            Imgproc.fillPoly(overlay, Collections.singletonList(mop), config.filledAreaColor);
            Core.addWeighted(overlay, config.filledAreaOpacity,
                    output, 1.0 - config.filledAreaOpacity, 0.0, output);
        } finally {
            mop.release();
            overlay.release();
        }
    }

    // =========================================================================================
    // Polynomial least-squares fit (x = c0·y^d + c1·y^(d-1) + … + cd)
    // =========================================================================================

    private static double[] fitPolynomial(List<Point> points, int degree) {
        int n    = points.size();
        int cols = degree + 1;
        Mat A    = new Mat(n, cols, CvType.CV_64F);
        Mat b    = new Mat(n, 1, CvType.CV_64F);
        Mat coef = new Mat();
        try {
            for (int i = 0; i < n; i++) {
                Point p = points.get(i);
                double y = p.y;
                double pow = 1.0;
                double[] row = new double[cols];
                for (int j = 0; j < cols; j++) { row[j] = pow; pow *= y; }
                for (int j = 0; j < cols; j++) A.put(i, j, row[cols - 1 - j]);
                b.put(i, 0, p.x);
            }
            if (!Core.solve(A, b, coef, Core.DECOMP_QR)) return null;
            double[] out = new double[cols];
            for (int i = 0; i < cols; i++) out[i] = coef.get(i, 0)[0];
            return out;
        } finally {
            A.release();
            b.release();
            coef.release();
        }
    }

    /** Horner's method — {@code coef[0]} is the highest-degree term. */
    private static double evalPoly(double[] coef, double y) {
        double r = coef[0];
        for (int i = 1; i < coef.length; i++) r = r * y + coef[i];
        return r;
    }
}
