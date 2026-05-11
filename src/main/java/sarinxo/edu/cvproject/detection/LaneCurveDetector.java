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
 * Detects individual lane-marking <b>segments</b> in a binary mask and draws a polynomial curve
 * over each one, strictly within the segment's own pixel extent.
 *
 * <h2>Design</h2>
 * Unlike a global histogram/sliding-window pipeline, this detector treats each connected white
 * region of the mask as an independent candidate. Each dashed mark is its own segment; a long
 * solid line is one big segment. Consequences:
 * <ul>
 *   <li>No bridging across pieces of dashed lines — only the dash that is actually visible
 *       is highlighted.</li>
 *   <li>No extrapolation to the top of the image — the curve is drawn only between the
 *       segment's own {@code yMin} and {@code yMax}.</li>
 *   <li>Pixels far from any plausible lane position are dropped as noise, not folded into a
 *       polynomial fit.</li>
 * </ul>
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Apply a configurable trapezoidal ROI to the mask.</li>
 *   <li>Restrict to the bottom band of the frame ({@link Config#segmentYMinRatio}). Only this
 *       region is searched, so the detector focuses on the road immediately ahead.</li>
 *   <li>Optionally warp to bird's-eye view.</li>
 *   <li>Find connected components ({@code connectedComponentsWithStats}). Drop components
 *       below {@link Config#minSegmentArea} pixels.</li>
 *   <li>For each surviving component, run {@link Imgproc#fitLine} to obtain a robust direction
 *       and discard horizontal/near-horizontal components ({@link Config#minSegmentVerticality}).</li>
 *   <li>Fit a polynomial of {@link Config#polynomialDegree} (2 by default) by least squares to
 *       the component's pixels.</li>
 *   <li>Compute the vanishing-point x by extrapolating each segment's line fit to {@code y=0}
 *       and averaging. The estimate is smoothed across frames. In BEV the centre is fixed at
 *       {@code bevWidth/2}.</li>
 *   <li>Filter segments whose base x is too far from the centre — those are noise.</li>
 *   <li>Pick the segment with the smallest negative {@code bottomX − centreX} as the left
 *       lane, and the smallest positive distance as the right lane.</li>
 *   <li>Draw each selected segment's polynomial from its {@code yMin} to its {@code yMax}.
 *       Nothing is drawn outside the segment's real pixel range.</li>
 * </ol>
 *
 * <h2>API</h2>
 * <pre>{@code
 * LaneMarkingMaskExtractor masker = new LaneMarkingMaskExtractor();
 * LaneCurveDetector       drawer = new LaneCurveDetector();
 * try {
 *     while (videoHasNextFrame()) {
 *         Mat frame  = readFrame();
 *         Mat mask   = masker.process(frame);
 *         Mat result = drawer.process(frame, mask);
 *         try { display(result); } finally { result.release(); }
 *         mask.release();
 *         frame.release();
 *     }
 * } finally { masker.release(); drawer.release(); }
 * }</pre>
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

        // -- Bottom-band restriction --
        public final double segmentYMinRatio;       // process only y >= H * this value

        // -- BEV --
        public final boolean bevEnabled;
        public final int     bevWidth;
        public final int     bevHeight;

        // -- Segment filtering --
        public final int    minSegmentArea;
        public final double minSegmentVerticality;             // |vy| of unit direction, 0..1
        public final double maxSegmentDistanceFromCenterRatio; // |dx|/W upper bound
        public final int    maxSegmentsToFit;

        // -- Polynomial --
        public final int polynomialDegree;
        public final int minPointsForFit;

        // -- Vanishing-point smoothing (image-space mode only) --
        public final double vpSmoothingAlpha;

        // -- Drawing --
        public final Scalar  leftColor;
        public final Scalar  rightColor;
        public final int     lineThickness;
        public final int     drawStepPixels;
        public final boolean drawFilledArea;
        public final Scalar  filledAreaColor;
        public final double  filledAreaOpacity;

        private Config(Builder b) {
            this.roiBottomY                        = b.roiBottomY;
            this.roiTopY                           = b.roiTopY;
            this.roiBottomHalfWidth                = b.roiBottomHalfWidth;
            this.roiTopHalfWidth                   = b.roiTopHalfWidth;
            this.segmentYMinRatio                  = b.segmentYMinRatio;
            this.bevEnabled                        = b.bevEnabled;
            this.bevWidth                          = b.bevWidth;
            this.bevHeight                         = b.bevHeight;
            this.minSegmentArea                    = b.minSegmentArea;
            this.minSegmentVerticality             = b.minSegmentVerticality;
            this.maxSegmentDistanceFromCenterRatio = b.maxSegmentDistanceFromCenterRatio;
            this.maxSegmentsToFit                  = b.maxSegmentsToFit;
            this.polynomialDegree                  = b.polynomialDegree;
            this.minPointsForFit                   = b.minPointsForFit;
            this.vpSmoothingAlpha                  = b.vpSmoothingAlpha;
            this.leftColor                         = b.leftColor;
            this.rightColor                        = b.rightColor;
            this.lineThickness                     = b.lineThickness;
            this.drawStepPixels                    = b.drawStepPixels;
            this.drawFilledArea                    = b.drawFilledArea;
            this.filledAreaColor                   = b.filledAreaColor;
            this.filledAreaOpacity                 = b.filledAreaOpacity;
        }

        public static Config  defaults() { return new Builder().build(); }
        public static Builder builder()  { return new Builder(); }

        public static final class Builder {
            // ROI
            private double  roiBottomY                        = 0.97;
            private double  roiTopY                           = 0.60;
            private double  roiBottomHalfWidth                = 0.50;
            private double  roiTopHalfWidth                   = 0.18;

            // Bottom band — by default the bottom 45% of the frame.
            private double  segmentYMinRatio                  = 0.55;

            // BEV
            private boolean bevEnabled                        = false;
            private int     bevWidth                          = 400;
            private int     bevHeight                         = 600;

            // Segment filters — calibrated for typical dashcam at 720p/1080p.
            private int     minSegmentArea                    = 60;
            private double  minSegmentVerticality             = 0.40;
            private double  maxSegmentDistanceFromCenterRatio = 0.35;
            private int     maxSegmentsToFit                  = 25;

            // Polynomial
            private int     polynomialDegree                  = 2;
            private int     minPointsForFit                   = 15;

            // VP smoothing
            private double  vpSmoothingAlpha                  = 0.25;

            // Drawing — BGR.
            private Scalar  leftColor                         = new Scalar(0,   255, 255);   // yellow
            private Scalar  rightColor                        = new Scalar(0,   0,   255);   // red
            private int     lineThickness                     = 6;
            private int     drawStepPixels                    = 3;
            private boolean drawFilledArea                    = false;
            private Scalar  filledAreaColor                   = new Scalar(0,   200, 0);     // green
            private double  filledAreaOpacity                 = 0.25;

            public Builder roiBottomY(double v)                        { this.roiBottomY = v; return this; }
            public Builder roiTopY(double v)                           { this.roiTopY = v; return this; }
            public Builder roiBottomHalfWidth(double v)                { this.roiBottomHalfWidth = v; return this; }
            public Builder roiTopHalfWidth(double v)                   { this.roiTopHalfWidth = v; return this; }
            public Builder segmentYMinRatio(double v)                  { this.segmentYMinRatio = v; return this; }
            public Builder bevEnabled(boolean v)                       { this.bevEnabled = v; return this; }
            public Builder bevWidth(int v)                             { this.bevWidth = v; return this; }
            public Builder bevHeight(int v)                            { this.bevHeight = v; return this; }
            public Builder minSegmentArea(int v)                       { this.minSegmentArea = v; return this; }
            public Builder minSegmentVerticality(double v)             { this.minSegmentVerticality = v; return this; }
            public Builder maxSegmentDistanceFromCenterRatio(double v) { this.maxSegmentDistanceFromCenterRatio = v; return this; }
            public Builder maxSegmentsToFit(int v)                     { this.maxSegmentsToFit = v; return this; }
            public Builder polynomialDegree(int v)                     { this.polynomialDegree = v; return this; }
            public Builder minPointsForFit(int v)                      { this.minPointsForFit = v; return this; }
            public Builder vpSmoothingAlpha(double v)                  { this.vpSmoothingAlpha = v; return this; }
            public Builder leftColor(Scalar v)                         { this.leftColor = v; return this; }
            public Builder rightColor(Scalar v)                        { this.rightColor = v; return this; }
            public Builder lineThickness(int v)                        { this.lineThickness = v; return this; }
            public Builder drawStepPixels(int v)                       { this.drawStepPixels = v; return this; }
            public Builder drawFilledArea(boolean v)                   { this.drawFilledArea = v; return this; }
            public Builder filledAreaColor(Scalar v)                   { this.filledAreaColor = v; return this; }
            public Builder filledAreaOpacity(double v)                 { this.filledAreaOpacity = v; return this; }

            public Config build() {
                if (roiBottomY <= roiTopY)
                    throw new IllegalArgumentException("roiBottomY must be > roiTopY");
                if (segmentYMinRatio < 0 || segmentYMinRatio > 1)
                    throw new IllegalArgumentException("segmentYMinRatio must be in [0,1]");
                if (polynomialDegree != 1 && polynomialDegree != 2 && polynomialDegree != 3)
                    throw new IllegalArgumentException("polynomialDegree must be 1, 2 or 3");
                if (minPointsForFit < polynomialDegree + 1)
                    throw new IllegalArgumentException("minPointsForFit must be > polynomialDegree");
                if (minSegmentVerticality < 0 || minSegmentVerticality > 1)
                    throw new IllegalArgumentException("minSegmentVerticality must be in [0,1]");
                if (vpSmoothingAlpha < 0 || vpSmoothingAlpha > 1)
                    throw new IllegalArgumentException("vpSmoothingAlpha must be in [0,1]");
                if (filledAreaOpacity < 0 || filledAreaOpacity > 1)
                    throw new IllegalArgumentException("filledAreaOpacity must be in [0,1]");
                Objects.requireNonNull(leftColor);
                Objects.requireNonNull(rightColor);
                return new Config(this);
            }
        }
    }

    // =========================================================================================
    // Internal model
    // =========================================================================================

    /** One detected lane-marking piece: a polynomial fit valid only inside [yMin, yMax]. */
    private static final class Segment {
        final double[] coef;          // polynomial coefficients, highest power first
        final int      yMin, yMax;
        final double   bottomX;       // x of the polynomial at yMax (closest point to camera)
        final double   convergenceX;  // x where the fitLine direction would reach y=0
        final int      pixelCount;

        Segment(double[] coef, int yMin, int yMax,
                double bottomX, double convergenceX, int pixelCount) {
            this.coef         = coef;
            this.yMin         = yMin;
            this.yMax         = yMax;
            this.bottomX      = bottomX;
            this.convergenceX = convergenceX;
            this.pixelCount   = pixelCount;
        }
    }

    // =========================================================================================
    // State
    // =========================================================================================

    private final Config config;

    private double vpEstimateX = Double.NaN;

    // Cached per-resolution geometry.
    private int     lastWidth        = -1;
    private int     lastHeight       = -1;
    private Point[] roiPolygonImage  = null;
    private Mat     origToBevMatrix  = null;
    private Mat     bevToOrigMatrix  = null;

    private boolean released = false;

    // -- Diagnostics --
    public static final class Diagnostics {
        public final int     componentsTotal;
        public final int     segmentsValid;
        public final double  centreX;
        public final boolean leftDetected;
        public final boolean rightDetected;

        Diagnostics(int ct, int sv, double cx, boolean ld, boolean rd) {
            this.componentsTotal = ct;
            this.segmentsValid   = sv;
            this.centreX         = cx;
            this.leftDetected    = ld;
            this.rightDetected   = rd;
        }

        @Override public String toString() {
            return String.format(
                    "Diagnostics{components=%d, validSegments=%d, centreX=%.1f, leftDetected=%s, rightDetected=%s}",
                    componentsTotal, segmentsValid, centreX, leftDetected, rightDetected);
        }
    }
    private Diagnostics lastDiagnostics = null;

    /** Counters from the most recent {@link #process} call, or {@code null} before the first. */
    public Diagnostics getLastDiagnostics() { return lastDiagnostics; }

    // =========================================================================================
    // Construction
    // =========================================================================================

    public LaneCurveDetector()              { this(Config.defaults()); }
    public LaneCurveDetector(Config config) { this.config = Objects.requireNonNull(config); }

    // =========================================================================================
    // Public API
    // =========================================================================================

    /**
     * Detect lane segments in {@code mask}, draw them on a clone of {@code frame}, and return
     * the clone. Each detected segment is drawn strictly within its own pixel extent.
     *
     * @param frame original BGR frame (CV_8UC3); not modified
     * @param mask  binary lane-marking mask of identical size (CV_8UC1)
     * @return new CV_8UC3 Mat with the segments overlaid; owned by the caller
     */
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

        // 1. ROI ------------------------------------------------------------------------------
        Mat roiMask = applyRoi(mask);

        // 2. Restrict to bottom band ----------------------------------------------------------
        int yCut = (int) Math.round(frame.height() * config.segmentYMinRatio);
        if (yCut > 0 && yCut < frame.height()) {
            Mat upper = roiMask.submat(0, yCut, 0, frame.width());
            upper.setTo(new Scalar(0));
            upper.release();
        }

        // 3. Optional BEV ---------------------------------------------------------------------
        Mat workingMask;
        int workingWidth, workingHeight;
        if (config.bevEnabled) {
            workingMask = new Mat();
            Imgproc.warpPerspective(roiMask, workingMask, origToBevMatrix,
                    new Size(config.bevWidth, config.bevHeight), Imgproc.INTER_NEAREST);
            workingWidth  = config.bevWidth;
            workingHeight = config.bevHeight;
        } else {
            workingMask   = roiMask;
            workingWidth  = frame.width();
            workingHeight = frame.height();
        }

        int componentsTotal = 0;
        List<Segment> segments;

        try {
            // 4. Connected components -> Segments ---------------------------------------------
            int[] countHolder = new int[]{0};
            segments = extractSegments(workingMask, workingWidth, workingHeight, countHolder);
            componentsTotal = countHolder[0];

            // 5. Centre estimation -----------------------------------------------------------
            double centreX = determineCentreX(segments, workingWidth);

            // 6. Reject segments too far from centre (noise) and select best left / right ----
            double maxDist = workingWidth * config.maxSegmentDistanceFromCenterRatio;
            Segment bestLeft  = null;
            Segment bestRight = null;
            double  leftDist  = Double.MAX_VALUE;
            double  rightDist = Double.MAX_VALUE;

            for (Segment s : segments) {
                double dx = s.bottomX - centreX;
                if (Math.abs(dx) > maxDist) continue;
                if (dx < 0 && -dx < leftDist)  { leftDist  = -dx; bestLeft  = s; }
                if (dx > 0 &&  dx < rightDist) { rightDist =  dx; bestRight = s; }
            }

            // 7. Draw --------------------------------------------------------------------------
            Mat output = frame.clone();
            drawSelected(output, bestLeft, bestRight, workingHeight);

            lastDiagnostics = new Diagnostics(
                    componentsTotal,
                    segments.size(),
                    centreX,
                    bestLeft  != null,
                    bestRight != null);

            return output;

        } finally {
            if (workingMask != roiMask) workingMask.release();
            roiMask.release();
        }
    }

    /** Reset all temporal state. Call between unrelated videos / scene cuts. */
    public void reset() {
        vpEstimateX = Double.NaN;
        lastWidth   = -1;
        lastHeight  = -1;
        releaseCachedGeometry();
    }

    /** Free all native resources. Safe to call multiple times. */
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
        vpEstimateX = Double.NaN;

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

        if (config.bevEnabled) {
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
    // Segment extraction
    // =========================================================================================

    private List<Segment> extractSegments(Mat workingMask, int width, int height,
                                          int[] componentsTotalOut) {
        Mat labels    = new Mat();
        Mat stats     = new Mat();
        Mat centroids = new Mat();
        Mat compMask  = new Mat();
        Mat lineMat   = new Mat();
        MatOfPoint   nonZero  = new MatOfPoint();
        MatOfPoint2f points2f = new MatOfPoint2f();

        List<Segment> segments = new ArrayList<>();

        try {
            int nLabels = Imgproc.connectedComponentsWithStats(
                    workingMask, labels, stats, centroids, 8, CvType.CV_32S);
            componentsTotalOut[0] = Math.max(0, nLabels - 1);

            // Collect candidate label indices sorted by area descending — bigger pieces first.
            List<int[]> ordered = new ArrayList<>(); // [label, area]
            for (int i = 1; i < nLabels; i++) {
                int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];
                if (area < config.minSegmentArea) continue;
                ordered.add(new int[]{i, area});
            }
            ordered.sort((a, b) -> Integer.compare(b[1], a[1]));

            int processed = 0;
            for (int[] entry : ordered) {
                if (processed >= config.maxSegmentsToFit) break;
                processed++;

                int label = entry[0];

                // Extract this component's pixels.
                Core.compare(labels, new Scalar(label), compMask, Core.CMP_EQ);
                Core.findNonZero(compMask, nonZero);
                if (nonZero.empty()) continue;
                Point[] pts = nonZero.toArray();
                if (pts.length < config.minPointsForFit) continue;

                // Robust line fit — direction + verticality test + extrapolation to y=0.
                points2f.fromArray(pts);
                Imgproc.fitLine(points2f, lineMat, Imgproc.DIST_L2, 0, 0.01, 0.01);
                double vx = lineMat.get(0, 0)[0];
                double vy = lineMat.get(1, 0)[0];
                double x0 = lineMat.get(2, 0)[0];
                double y0 = lineMat.get(3, 0)[0];

                // (vx, vy) is a unit vector along the line direction, so |vy| in [0, 1].
                if (Math.abs(vy) < config.minSegmentVerticality) continue;

                double convergenceX = x0 + (0 - y0) * vx / vy;

                // Polynomial fit on the segment's own pixels.
                double[] coef = fitPolynomial(pts, config.polynomialDegree);
                if (coef == null) continue;

                // True yMin / yMax from pixel data.
                int yMin = Integer.MAX_VALUE;
                int yMax = Integer.MIN_VALUE;
                for (Point p : pts) {
                    int yi = (int) p.y;
                    if (yi < yMin) yMin = yi;
                    if (yi > yMax) yMax = yi;
                }

                double bottomX = evalPoly(coef, yMax);

                segments.add(new Segment(coef, yMin, yMax, bottomX, convergenceX, pts.length));
            }
        } finally {
            labels.release();
            stats.release();
            centroids.release();
            compMask.release();
            lineMat.release();
            nonZero.release();
            points2f.release();
        }
        return segments;
    }

    // =========================================================================================
    // Centre estimation
    // =========================================================================================

    private double determineCentreX(List<Segment> segments, int workingWidth) {
        if (config.bevEnabled) {
            // BEV — lanes are parallel; centre is geometric centre.
            return workingWidth * 0.5;
        }

        // Image space — average each segment's convergence point and smooth across frames.
        if (!segments.isEmpty()) {
            double sum   = 0;
            int    count = 0;
            for (Segment s : segments) {
                if (!Double.isFinite(s.convergenceX)) continue;
                if (s.convergenceX < 0 || s.convergenceX > workingWidth) continue;
                sum += s.convergenceX;
                count++;
            }
            if (count > 0) {
                double frameVp = sum / count;
                if (Double.isNaN(vpEstimateX)) {
                    vpEstimateX = frameVp;
                } else {
                    double a = config.vpSmoothingAlpha;
                    vpEstimateX = a * frameVp + (1 - a) * vpEstimateX;
                }
            }
        }
        return Double.isNaN(vpEstimateX) ? workingWidth * 0.5 : vpEstimateX;
    }

    // =========================================================================================
    // Polynomial utilities
    // =========================================================================================

    /** Fits <code>x = c0·y^d + c1·y^(d-1) + … + cd</code> by least squares. */
    private static double[] fitPolynomial(Point[] points, int degree) {
        int n    = points.length;
        int cols = degree + 1;
        Mat A    = new Mat(n, cols, CvType.CV_64F);
        Mat b    = new Mat(n, 1, CvType.CV_64F);
        Mat coef = new Mat();
        try {
            for (int i = 0; i < n; i++) {
                Point p = points[i];
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

    // =========================================================================================
    // Drawing
    // =========================================================================================

    private void drawSelected(Mat output, Segment left, Segment right, int workingHeight) {
        if (left == null && right == null) return;

        Point[] leftPts  = (left  != null) ? curvePointsImage(left,  workingHeight) : null;
        Point[] rightPts = (right != null) ? curvePointsImage(right, workingHeight) : null;

        if (config.drawFilledArea && leftPts != null && rightPts != null) {
            drawFilledArea(output, leftPts, rightPts);
        }
        if (leftPts  != null) drawPolyline(output, leftPts,  config.leftColor);
        if (rightPts != null) drawPolyline(output, rightPts, config.rightColor);
    }

    /** Samples a segment's polynomial only between its yMin and yMax. */
    private Point[] curvePointsImage(Segment seg, int workingHeight) {
        int step = Math.max(1, config.drawStepPixels);
        List<Point> samples = new ArrayList<>();
        for (int y = seg.yMin; y <= seg.yMax; y += step) {
            double x = evalPoly(seg.coef, y);
            samples.add(new Point(x, y));
        }
        // Ensure the last endpoint at yMax is included so the curve reaches the camera-side end.
        if (samples.isEmpty() || samples.get(samples.size() - 1).y < seg.yMax) {
            samples.add(new Point(evalPoly(seg.coef, seg.yMax), seg.yMax));
        }
        if (samples.size() < 2) return null;

        if (!config.bevEnabled) {
            return samples.toArray(new Point[0]);
        }

        // BEV -> image-space.
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
}