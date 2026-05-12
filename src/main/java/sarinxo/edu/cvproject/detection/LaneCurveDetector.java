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
 *   <li>Sum the bottom strip of BEV column-wise into a 1D histogram. The strongest peak in
 *       the left half is the left lane base, the strongest in the right half is the right
 *       base. Because the adaptive ROI has already centred the road, the halves correspond
 *       to physical left/right of the ego lane.</li>
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

        // -- Adaptive ROI centre (estimates road centre from the mask each frame) --
        public final boolean adaptiveCenterEnabled;
        public final double  centerEstimateAlpha;            // EMA factor for the per-frame estimate
        public final double  centerStripHeightRatio;         // fraction of frame height sampled at the bottom
        public final double  centerEstimateSearchMargin;     // peaks must lie in [margin, 1−margin] of frame width
        public final double  centerEstimateMinSeparationRatio; // min |xL − xR| / frameWidth between peaks
        public final int     centerEstimateMinPeakSum;       // min column-sum to accept a peak

        // -- BEV target size --
        public final int bevWidth;
        public final int bevHeight;

        // -- Histogram base detection --
        public final int    histogramMinPeakSum;       // min column-sum to accept a base peak
        public final double histogramBottomFraction;   // fraction of BEV bottom used for the histogram

        // -- Output validation (segmented drawing) --
        public final int    outputSupportHalfWidthPx;  // half-width of the mask tube checked at each y
        public final int    outputMinRunLength;        // min consecutive supported y's to emit as one segment

        // -- Parallel-lane hypothesis fallback --
        public final double laneWidthAlpha;            // EMA factor for the learned BEV lane width
        public final double hypothesisMinSupportFraction;  // fraction of y-range that must show mask support to accept a hypothesised lane

        // -- Sliding window --
        public final int    slidingWindowCount;
        public final int    slidingWindowMarginX;        // half-width in BEV pixels
        public final int    slidingWindowMinPixels;      // min pixels in window to fit a local line
        public final double maxAngleFromVerticalDeg;     // discard a window whose local tangent leans more than this from BEV vertical
        public final double maxAngleStepDeg;             // discard a window whose tangent differs from the previous window's by more than this
        public final double windowLaneHalfWidthPx;       // perpendicular half-width: pixels farther than this from the local axis are noise

        // -- Polynomial --
        public final int polynomialDegree;
        public final int minPointsForFit;

        // -- Outlier rejection --
        public final double residualThresholdPx;       // BEV pixels: |x − poly(y)| over which a pixel is dropped
        public final int    refinementIterations;      // number of refit passes after initial fit

        // -- Drawing --
        public final Scalar  leftColor;
        public final Scalar  rightColor;
        public final int     lineThickness;
        public final int     drawStepPixels;

        private Config(Builder b) {
            this.roiBottomY              = b.roiBottomY;
            this.roiTopY                 = b.roiTopY;
            this.roiBottomHalfWidth      = b.roiBottomHalfWidth;
            this.roiTopHalfWidth         = b.roiTopHalfWidth;
            this.adaptiveCenterEnabled           = b.adaptiveCenterEnabled;
            this.centerEstimateAlpha             = b.centerEstimateAlpha;
            this.centerStripHeightRatio          = b.centerStripHeightRatio;
            this.centerEstimateSearchMargin      = b.centerEstimateSearchMargin;
            this.centerEstimateMinSeparationRatio = b.centerEstimateMinSeparationRatio;
            this.centerEstimateMinPeakSum        = b.centerEstimateMinPeakSum;
            this.bevWidth                = b.bevWidth;
            this.bevHeight               = b.bevHeight;
            this.histogramMinPeakSum     = b.histogramMinPeakSum;
            this.histogramBottomFraction = b.histogramBottomFraction;
            this.outputSupportHalfWidthPx     = b.outputSupportHalfWidthPx;
            this.outputMinRunLength           = b.outputMinRunLength;
            this.laneWidthAlpha               = b.laneWidthAlpha;
            this.hypothesisMinSupportFraction = b.hypothesisMinSupportFraction;
            this.slidingWindowCount       = b.slidingWindowCount;
            this.slidingWindowMarginX     = b.slidingWindowMarginX;
            this.slidingWindowMinPixels   = b.slidingWindowMinPixels;
            this.maxAngleFromVerticalDeg  = b.maxAngleFromVerticalDeg;
            this.maxAngleStepDeg          = b.maxAngleStepDeg;
            this.windowLaneHalfWidthPx    = b.windowLaneHalfWidthPx;
            this.polynomialDegree        = b.polynomialDegree;
            this.minPointsForFit         = b.minPointsForFit;
            this.residualThresholdPx     = b.residualThresholdPx;
            this.refinementIterations    = b.refinementIterations;
            this.leftColor               = b.leftColor;
            this.rightColor              = b.rightColor;
            this.lineThickness           = b.lineThickness;
            this.drawStepPixels          = b.drawStepPixels;
        }

        public static Config  defaults() { return new Builder().build(); }
        public static Builder builder()  { return new Builder(); }

        public static final class Builder {
            // ROI — narrowed to roughly the visible ego-lane footprint. A tight trapezoid is
            // critical: any wall, barrier or roadside foliage left inside the ROI projects to
            // BEV as a strong vertical-looking cluster that the histogram will treat as a lane
            // base and the sliding window will then trace as if it were real paint.
            private double  roiBottomY              = 0.95;
            private double  roiTopY                 = 0.62;
            private double  roiBottomHalfWidth      = 0.35;
            private double  roiTopHalfWidth         = 0.10;

            // Adaptive centre — automatic horizontal offset of the trapezoid based on where the
            // two strongest lane bases appear at the bottom of the mask. Handles dashcams that
            // are not mounted on the car's longitudinal centreline.
            private boolean adaptiveCenterEnabled            = true;
            private double  centerEstimateAlpha              = 0.20;
            private double  centerStripHeightRatio           = 0.08;
            // Peaks must live in the interior 80% of the frame — pure edges almost always
            // contain barrier / foliage clutter, not lane paint.
            private double  centerEstimateSearchMargin       = 0.10;
            private double  centerEstimateMinSeparationRatio = 0.15;
            private int     centerEstimateMinPeakSum         = 10 * 255;

            // BEV resolution — independent of input frame size.
            private int     bevWidth                = 400;
            private int     bevHeight               = 600;

            // Histogram peak threshold. Mask pixels are 0/255, so a column-sum value of
            // (N × 255) means N white pixels accumulated in that column of the histogram band.
            // 25 pixels is enough to dismiss isolated noise but accept short dashes.
            private int     histogramMinPeakSum     = 25 * 255;

            // Bottom 55% of BEV used for the histogram. Wider than 1/3 so a far-away dashed
            // line, which is still in the middle of BEV, also contributes to its peak.
            private double  histogramBottomFraction = 0.55;

            // Output validation. Walk along each fitted polynomial in BEV and emit only the
            // y-intervals where the mask actually has paint within ±outputSupportHalfWidthPx
            // of the curve. outputMinRunLength filters out specks; intervals shorter than
            // this number of BEV pixels are dropped so single hot pixels do not become dashes.
            private int     outputSupportHalfWidthPx     = 18;
            private int     outputMinRunLength           = 8;

            // Lane-width EMA — smooths the BEV distance between confirmed left and right
            // lanes across frames, then used as a parallel-lane hypothesis when one side
            // becomes too weak to detect on its own.
            private double  laneWidthAlpha               = 0.20;

            // A hypothesised lane (copy of the detected one shifted by learned lane width)
            // is accepted only if at least this fraction of its y-range shows mask support.
            // 0.20 = 20%: enough for a sparse dashed line, strict enough to reject noise.
            private double  hypothesisMinSupportFraction = 0.20;

            // Sliding window — 9 vertical slices is the standard Udacity-style setup.
            // Margin tightened so the window stays close to the lane and resists pull
            // from neighbouring objects (wheels, signs).
            private int    slidingWindowCount       = 9;
            private int    slidingWindowMarginX     = 45;
            private int    slidingWindowMinPixels   = 20;

            // Per-window geometric guards — derived from the physical assumption that lane
            // marking is locally near-vertical in BEV and changes direction smoothly.
            // 35° lets sharp curves through; 15° per ~67-px window forbids implausible jumps
            // caused by a noise cluster pulling the local line fit.
            private double maxAngleFromVerticalDeg  = 35.0;
            private double maxAngleStepDeg          = 15.0;
            private double windowLaneHalfWidthPx    = 15.0;

            // Cubic polynomial — captures S-curves and roundabouts; quadratic is too rigid
            // to trace strong curvature faithfully.
            private int     polynomialDegree        = 3;
            private int     minPointsForFit         = 25;

            // RANSAC-style outlier rejection. After the initial fit, pixels whose horizontal
            // distance to the polynomial exceeds residualThresholdPx are dropped, then the
            // polynomial is refit. Two passes are sufficient to converge to the true lane
            // when wheels or shadows briefly enter the search window.
            private double  residualThresholdPx     = 20.0;
            private int     refinementIterations    = 2;

            // Drawing — BGR. Yellow left, red right.
            private Scalar  leftColor               = new Scalar(0,   255, 255);
            private Scalar  rightColor              = new Scalar(0,   0,   255);
            private int     lineThickness           = 6;
            private int     drawStepPixels          = 2;

            public Builder roiBottomY(double v)              { this.roiBottomY = v; return this; }
            public Builder roiTopY(double v)                 { this.roiTopY = v; return this; }
            public Builder roiBottomHalfWidth(double v)      { this.roiBottomHalfWidth = v; return this; }
            public Builder roiTopHalfWidth(double v)         { this.roiTopHalfWidth = v; return this; }
            public Builder adaptiveCenterEnabled(boolean v)            { this.adaptiveCenterEnabled = v; return this; }
            public Builder centerEstimateAlpha(double v)               { this.centerEstimateAlpha = v; return this; }
            public Builder centerStripHeightRatio(double v)            { this.centerStripHeightRatio = v; return this; }
            public Builder centerEstimateSearchMargin(double v)        { this.centerEstimateSearchMargin = v; return this; }
            public Builder centerEstimateMinSeparationRatio(double v)  { this.centerEstimateMinSeparationRatio = v; return this; }
            public Builder centerEstimateMinPeakSum(int v)             { this.centerEstimateMinPeakSum = v; return this; }
            public Builder bevWidth(int v)                   { this.bevWidth = v; return this; }
            public Builder bevHeight(int v)                  { this.bevHeight = v; return this; }
            public Builder histogramMinPeakSum(int v)        { this.histogramMinPeakSum = v; return this; }
            public Builder histogramBottomFraction(double v) { this.histogramBottomFraction = v; return this; }
            public Builder outputSupportHalfWidthPx(int v)            { this.outputSupportHalfWidthPx = v; return this; }
            public Builder outputMinRunLength(int v)                  { this.outputMinRunLength = v; return this; }
            public Builder laneWidthAlpha(double v)                   { this.laneWidthAlpha = v; return this; }
            public Builder hypothesisMinSupportFraction(double v)     { this.hypothesisMinSupportFraction = v; return this; }
            public Builder slidingWindowCount(int v)         { this.slidingWindowCount = v; return this; }
            public Builder slidingWindowMarginX(int v)       { this.slidingWindowMarginX = v; return this; }
            public Builder slidingWindowMinPixels(int v)     { this.slidingWindowMinPixels = v; return this; }
            public Builder maxAngleFromVerticalDeg(double v) { this.maxAngleFromVerticalDeg = v; return this; }
            public Builder maxAngleStepDeg(double v)         { this.maxAngleStepDeg = v; return this; }
            public Builder windowLaneHalfWidthPx(double v)   { this.windowLaneHalfWidthPx = v; return this; }
            public Builder polynomialDegree(int v)           { this.polynomialDegree = v; return this; }
            public Builder minPointsForFit(int v)            { this.minPointsForFit = v; return this; }
            public Builder residualThresholdPx(double v)     { this.residualThresholdPx = v; return this; }
            public Builder refinementIterations(int v)       { this.refinementIterations = v; return this; }
            public Builder leftColor(Scalar v)               { this.leftColor = v; return this; }
            public Builder rightColor(Scalar v)              { this.rightColor = v; return this; }
            public Builder lineThickness(int v)              { this.lineThickness = v; return this; }
            public Builder drawStepPixels(int v)             { this.drawStepPixels = v; return this; }

            public Config build() {
                if (roiBottomY <= roiTopY)
                    throw new IllegalArgumentException("roiBottomY must be > roiTopY");
                if (bevWidth < 50 || bevHeight < 50)
                    throw new IllegalArgumentException("BEV dimensions too small");
                if (slidingWindowCount < 3)
                    throw new IllegalArgumentException("slidingWindowCount must be >= 3");
                if (slidingWindowMarginX <= 0)
                    throw new IllegalArgumentException("slidingWindowMarginX must be > 0");
                if (polynomialDegree < 1 || polynomialDegree > 3)
                    throw new IllegalArgumentException("polynomialDegree must be 1, 2 or 3");
                if (minPointsForFit < polynomialDegree + 1)
                    throw new IllegalArgumentException("minPointsForFit must be > polynomialDegree");
                if (histogramBottomFraction <= 0 || histogramBottomFraction > 1)
                    throw new IllegalArgumentException("histogramBottomFraction must be in (0,1]");
                if (outputSupportHalfWidthPx <= 0)
                    throw new IllegalArgumentException("outputSupportHalfWidthPx must be > 0");
                if (outputMinRunLength < 1)
                    throw new IllegalArgumentException("outputMinRunLength must be >= 1");
                if (laneWidthAlpha <= 0 || laneWidthAlpha > 1)
                    throw new IllegalArgumentException("laneWidthAlpha must be in (0,1]");
                if (hypothesisMinSupportFraction < 0 || hypothesisMinSupportFraction > 1)
                    throw new IllegalArgumentException("hypothesisMinSupportFraction must be in [0,1]");
                if (centerEstimateAlpha <= 0 || centerEstimateAlpha > 1)
                    throw new IllegalArgumentException("centerEstimateAlpha must be in (0,1]");
                if (centerStripHeightRatio <= 0 || centerStripHeightRatio > 1)
                    throw new IllegalArgumentException("centerStripHeightRatio must be in (0,1]");
                if (centerEstimateSearchMargin < 0 || centerEstimateSearchMargin >= 0.5)
                    throw new IllegalArgumentException("centerEstimateSearchMargin must be in [0,0.5)");
                if (centerEstimateMinSeparationRatio < 0 || centerEstimateMinSeparationRatio > 1)
                    throw new IllegalArgumentException("centerEstimateMinSeparationRatio must be in [0,1]");
                if (maxAngleFromVerticalDeg <= 0 || maxAngleFromVerticalDeg > 89)
                    throw new IllegalArgumentException("maxAngleFromVerticalDeg must be in (0,89]");
                if (maxAngleStepDeg <= 0 || maxAngleStepDeg > 89)
                    throw new IllegalArgumentException("maxAngleStepDeg must be in (0,89]");
                if (windowLaneHalfWidthPx <= 0)
                    throw new IllegalArgumentException("windowLaneHalfWidthPx must be > 0");
                if (residualThresholdPx <= 0)
                    throw new IllegalArgumentException("residualThresholdPx must be > 0");
                if (refinementIterations < 0)
                    throw new IllegalArgumentException("refinementIterations must be >= 0");
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
    private double  lastCenterRatio  = Double.NaN;
    private Point[] roiPolygonImage  = null;
    private Mat     origToBevMatrix  = null;
    private Mat     bevToOrigMatrix  = null;

    // EMA estimate of the road horizontal centre, as a fraction of frame width.
    // Updated each frame from the bottom-strip histogram; NaN means "no estimate yet".
    private double  currentCenterXRatio = Double.NaN;

    // EMA estimate of lane width in BEV pixels, learned from frames where both lanes were
    // detected. Drives the parallel-lane hypothesis when only one side is found.
    private double  learnedLaneWidthBev = Double.NaN;

    private boolean released = false;

    // -- Diagnostics --
    public static final class Diagnostics {
        public final int     leftBaseX;
        public final int     rightBaseX;
        public final int     leftPixels;
        public final int     rightPixels;
        public final boolean leftDetected;
        public final boolean rightDetected;
        public final double  centerXRatio;       // adaptive ROI centre (image-space, fraction of width)

        Diagnostics(int lb, int rb, int lp, int rp, boolean ld, boolean rd, double cxr) {
            this.leftBaseX     = lb;
            this.rightBaseX    = rb;
            this.leftPixels    = lp;
            this.rightPixels   = rp;
            this.leftDetected  = ld;
            this.rightDetected = rd;
            this.centerXRatio  = cxr;
        }

        @Override public String toString() {
            return String.format(
                    "Diagnostics{leftBaseX=%d, rightBaseX=%d, leftPixels=%d, rightPixels=%d, "
                    + "leftDetected=%s, rightDetected=%s, centerXRatio=%.3f}",
                    leftBaseX, rightBaseX, leftPixels, rightPixels,
                    leftDetected, rightDetected, centerXRatio);
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

        if (config.adaptiveCenterEnabled) {
            double frameCenterRatio = estimateRoadCenterRatio(mask);
            if (!Double.isNaN(frameCenterRatio)) {
                if (Double.isNaN(currentCenterXRatio)) {
                    currentCenterXRatio = frameCenterRatio;            // bootstrap from first valid frame
                } else {
                    double a = config.centerEstimateAlpha;
                    currentCenterXRatio = (1 - a) * currentCenterXRatio + a * frameCenterRatio;
                }
            }
        }
        double centerRatio = Double.isNaN(currentCenterXRatio) ? 0.5 : currentCenterXRatio;
        ensureGeometry(frame.width(), frame.height(), centerRatio);

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

            double[] leftCoef  = robustFit(leftPixels);
            double[] rightCoef = robustFit(rightPixels);

            // Learn lane width from frames where both lanes are detected — used as a
            // hypothesis seed on frames where one side is missing.
            updateLearnedLaneWidth(leftCoef, rightCoef);

            // Parallel-lane hypothesis. If one side has no polynomial but the other does
            // AND we have a learned width, propose a copy of the strong curve shifted by
            // the width. Accept only if it lines up with paint in the mask.
            boolean leftIsHypothesis  = false;
            boolean rightIsHypothesis = false;
            if (leftCoef == null && rightCoef != null && !Double.isNaN(learnedLaneWidthBev)) {
                double[] hypo = shiftPolynomial(rightCoef, -learnedLaneWidthBev);
                if (hypothesisIsSupported(bev, hypo)) {
                    leftCoef = hypo;
                    leftIsHypothesis = true;
                }
            } else if (rightCoef == null && leftCoef != null && !Double.isNaN(learnedLaneWidthBev)) {
                double[] hypo = shiftPolynomial(leftCoef, +learnedLaneWidthBev);
                if (hypothesisIsSupported(bev, hypo)) {
                    rightCoef = hypo;
                    rightIsHypothesis = true;
                }
            }

            Mat output = frame.clone();
            drawLanes(output, bev, leftCoef, rightCoef, leftPixels, rightPixels,
                      leftIsHypothesis, rightIsHypothesis);

            lastDiagnostics = new Diagnostics(
                    leftBaseX, rightBaseX,
                    leftPixels.size(), rightPixels.size(),
                    leftCoef != null, rightCoef != null,
                    centerRatio);

            return output;
        } finally {
            bev.release();
            roiMask.release();
        }
    }

    public void reset() {
        lastWidth           = -1;
        lastHeight          = -1;
        lastCenterRatio     = Double.NaN;
        currentCenterXRatio = Double.NaN;
        learnedLaneWidthBev = Double.NaN;
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

    /**
     * Rebuild the ROI trapezoid and perspective matrices. The trapezoid is asymmetric: its
     * bottom centre tracks the estimated road centre (so the ROI follows a laterally-offset
     * dashcam), while its top centre stays at the image centre — the vanishing point of any
     * forward-facing camera is on the camera's optical axis, regardless of lateral mount
     * position. Rebuilding happens only when frame size or centre changed materially.
     */
    private void ensureGeometry(int w, int h, double centerXRatio) {
        if (w == lastWidth && h == lastHeight
                && Math.abs(centerXRatio - lastCenterRatio) < 0.003) return;
        releaseCachedGeometry();

        double cxBottom    = w * centerXRatio;
        double cxTop       = w * 0.5;
        double bottomY     = h * config.roiBottomY;
        double topY        = h * config.roiTopY;
        double bottomHalf  = w * config.roiBottomHalfWidth;
        double topHalf     = w * config.roiTopHalfWidth;

        roiPolygonImage = new Point[] {
                new Point(cxBottom - bottomHalf, bottomY),
                new Point(cxTop    - topHalf,    topY),
                new Point(cxTop    + topHalf,    topY),
                new Point(cxBottom + bottomHalf, bottomY)
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

        lastWidth       = w;
        lastHeight      = h;
        lastCenterRatio = centerXRatio;
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
    // Adaptive ROI centre — estimate where the road is centred in the current frame
    // =========================================================================================

    /**
     * Estimate the road's horizontal centre from a bottom strip of the mask, in image space.
     * Both lane lines are at their most separated and most reliable near the camera, so we
     * sum a thin band at the bottom of the frame into a column histogram and pick the two
     * strongest peaks inside the interior of the frame (edges typically contain barrier /
     * foliage clutter). The mid-point of those peaks is the estimate. Returns {@code NaN}
     * when fewer than two usable peaks are present — the caller keeps the previous estimate.
     */
    private double estimateRoadCenterRatio(Mat mask) {
        int stripH = (int) Math.max(5, Math.round(mask.rows() * config.centerStripHeightRatio));
        Mat strip  = mask.submat(mask.rows() - stripH, mask.rows(), 0, mask.cols());
        Mat hMat   = new Mat();
        try {
            Core.reduce(strip, hMat, 0, Core.REDUCE_SUM, CvType.CV_32S);
            int w = hMat.cols();
            int[] hist = new int[w];
            hMat.get(0, 0, hist);

            int searchLow  = (int) (w * config.centerEstimateSearchMargin);
            int searchHigh = (int) (w * (1.0 - config.centerEstimateSearchMargin));
            int minSep     = (int) (w * config.centerEstimateMinSeparationRatio);
            int minPeakVal = config.centerEstimateMinPeakSum;

            int max1V = -1, max1X = -1;
            for (int x = searchLow; x < searchHigh; x++) {
                if (hist[x] > max1V) { max1V = hist[x]; max1X = x; }
            }
            if (max1V < minPeakVal) return Double.NaN;

            int max2V = -1, max2X = -1;
            for (int x = searchLow; x < searchHigh; x++) {
                if (Math.abs(x - max1X) < minSep) continue;
                if (hist[x] > max2V) { max2V = hist[x]; max2X = x; }
            }
            if (max2V < minPeakVal) return Double.NaN;

            return ((max1X + max2X) / 2.0) / w;
        } finally {
            strip.release();
            hMat.release();
        }
    }

    // =========================================================================================
    // Base detection — find left and right lane starting x by histogram peaks
    // =========================================================================================

    /**
     * Returns {@code {leftBaseX, rightBaseX}} in BEV coordinates. Either value may be
     * {@code -1} when that side cannot be located. The two halves of BEV are searched
     * independently — the adaptive ROI has already centred the road horizontally in BEV,
     * so the strongest left-half peak is the left lane base and the strongest right-half
     * peak is the right. This avoids the pathological case where NMS placed both bases
     * on opposite sides of a single wide noise cluster.
     */
    private int[] findBases(Mat bev) {
        int bandStart = (int) Math.round(bev.rows() * (1.0 - config.histogramBottomFraction));
        Mat band      = bev.submat(bandStart, bev.rows(), 0, bev.cols());
        Mat hMat      = new Mat();
        try {
            Core.reduce(band, hMat, 0, Core.REDUCE_SUM, CvType.CV_32S);
            int w = hMat.cols();
            int[] hist = new int[w];
            hMat.get(0, 0, hist);

            int mid    = w / 2;
            int minVal = config.histogramMinPeakSum;

            int leftV  = -1, leftX  = -1;
            for (int x = 0; x < mid; x++) {
                if (hist[x] > leftV) { leftV = hist[x]; leftX = x; }
            }
            int rightV = -1, rightX = -1;
            for (int x = mid; x < w; x++) {
                if (hist[x] > rightV) { rightV = hist[x]; rightX = x; }
            }

            return new int[]{
                    leftV  >= minVal ? leftX  : -1,
                    rightV >= minVal ? rightX : -1
            };
        } finally {
            band.release();
            hMat.release();
        }
    }

    // =========================================================================================
    // Sliding window — collect lane pixels by walking upward from the base
    // =========================================================================================

    /**
     * Walk a sliding window upward from {@code baseX} through the BEV. Each window does three
     * things informed by lane geometry:
     * <ol>
     *   <li>Fits a local line (Huber-robust) to its pixels — the line's direction is the lane's
     *       local tangent.</li>
     *   <li>Checks that the tangent leans no more than {@code maxAngleFromVerticalDeg} from
     *       BEV vertical, and that it changes by no more than {@code maxAngleStepDeg} since
     *       the previous accepted window. A sharp kink is structurally impossible for real
     *       paint and is therefore a noise cluster.</li>
     *   <li>Keeps only pixels lying within {@code windowLaneHalfWidthPx} of that local
     *       tangent — the lane is a thin band; pixels off to one side are wheel/shadow noise
     *       that happened to fall into the rectangle.</li>
     * </ol>
     * The next window's search column is predicted by extrapolating the tangent, not by
     * shifting to the centroid — so an outlier-poisoned window cannot drag the trajectory.
     */
    private List<Point> slidingWindowSearch(Mat bev, int baseX) {
        int H      = bev.rows();
        int W      = bev.cols();
        int nWin   = config.slidingWindowCount;
        int winH   = H / nWin;
        int margin = config.slidingWindowMarginX;
        int minPix = config.slidingWindowMinPixels;

        double maxVerticalRad = Math.toRadians(config.maxAngleFromVerticalDeg);
        double maxStepRad     = Math.toRadians(config.maxAngleStepDeg);
        double halfWidth      = config.windowLaneHalfWidthPx;

        double currentX     = baseX;
        double currentSlope = 0.0;             // dx/dy from last accepted window — 0 = vertical
        double prevAngle    = Double.NaN;      // signed angle from vertical (rad)
        int    prevMidY     = -1;

        List<Point>   collected = new ArrayList<>();
        MatOfPoint    nonZero   = new MatOfPoint();
        MatOfPoint2f  points2f  = new MatOfPoint2f();
        Mat           lineMat   = new Mat();
        try {
            for (int wIdx = 0; wIdx < nWin; wIdx++) {
                int yLow  = Math.max(0, H - (wIdx + 1) * winH);
                int yHigh = H - wIdx * winH;
                if (yLow >= yHigh) continue;
                int midY  = (yLow + yHigh) / 2;

                // Predict this window's centre x by extrapolating the last accepted tangent.
                double predictedX = (prevMidY < 0)
                        ? currentX
                        : currentX + currentSlope * (midY - prevMidY);

                int xLow  = (int) Math.max(0, Math.round(predictedX - margin));
                int xHigh = (int) Math.min(W, Math.round(predictedX + margin));
                if (xLow >= xHigh) continue;

                Mat slice = bev.submat(yLow, yHigh, xLow, xHigh);
                try {
                    Core.findNonZero(slice, nonZero);
                    int n = (int) nonZero.total();
                    if (n < minPix) continue;                   // too sparse — keep predicted state

                    Point[] local = nonZero.toArray();
                    Point[] absPts = new Point[local.length];
                    for (int i = 0; i < local.length; i++) {
                        absPts[i] = new Point(xLow + local[i].x, yLow + local[i].y);
                    }

                    // Local line fit — Huber resists pull from off-axis clusters within the window.
                    points2f.fromArray(absPts);
                    Imgproc.fitLine(points2f, lineMat, Imgproc.DIST_HUBER, 0, 0.01, 0.01);
                    double vx = lineMat.get(0, 0)[0];
                    double vy = lineMat.get(1, 0)[0];
                    double x0 = lineMat.get(2, 0)[0];
                    double y0 = lineMat.get(3, 0)[0];

                    // Normalise direction so vy is non-negative (line "points forward" in BEV).
                    if (vy < 0) { vx = -vx; vy = -vy; }

                    // Geometric guard 1: local tangent must be near-vertical.
                    double angle = Math.atan2(vx, vy);            // 0 = vertical, ±π/2 = horizontal
                    if (Math.abs(angle) > maxVerticalRad) continue;

                    // Geometric guard 2: tangent must not flip relative to the previous window.
                    if (!Double.isNaN(prevAngle) && Math.abs(angle - prevAngle) > maxStepRad) continue;

                    // Geometric guard 3: keep only pixels within a thin band along the local axis.
                    // Perpendicular distance to the line (x0, y0) + t*(vx, vy):
                    //     d = |(p.x − x0) · vy − (p.y − y0) · vx|   (since (vx,vy) is unit length)
                    int kept = 0;
                    for (Point p : absPts) {
                        double d = Math.abs((p.x - x0) * vy - (p.y - y0) * vx);
                        if (d <= halfWidth) {
                            collected.add(p);
                            kept++;
                        }
                    }
                    if (kept == 0) continue;

                    // Update state from the accepted tangent.
                    currentX     = x0 + (midY - y0) * vx / vy;
                    currentSlope = vx / vy;
                    prevAngle    = angle;
                    prevMidY     = midY;
                } finally {
                    slice.release();
                }
            }
        } finally {
            nonZero.release();
            points2f.release();
            lineMat.release();
        }
        return collected;
    }

    // =========================================================================================
    // Drawing — sample polynomial in BEV, warp back to image, draw polylines
    // =========================================================================================

    /**
     * Draws each side as one or more separate polyline segments — only the y-intervals where
     * the BEV mask actually carries paint within ±{@code outputSupportHalfWidthPx} of the
     * polynomial are emitted. A dashed lane naturally renders as a sequence of short
     * polylines (one per dash); a solid lane renders as a single long polyline. No paint —
     * no segment, no extrapolation.
     */
    private void drawLanes(Mat output, Mat bev,
                           double[] leftCoef, double[] rightCoef,
                           List<Point> leftPixels, List<Point> rightPixels,
                           boolean leftIsHypothesis, boolean rightIsHypothesis) {
        if (leftCoef != null) {
            int yMin = leftIsHypothesis ? 0 : pixelYMin(leftPixels);
            drawValidatedSegments(output, bev, leftCoef, yMin, config.bevHeight - 1, config.leftColor);
        }
        if (rightCoef != null) {
            int yMin = rightIsHypothesis ? 0 : pixelYMin(rightPixels);
            drawValidatedSegments(output, bev, rightCoef, yMin, config.bevHeight - 1, config.rightColor);
        }
    }

    private int pixelYMin(List<Point> pts) {
        int y = Integer.MAX_VALUE;
        for (Point p : pts) if (p.y < y) y = (int) p.y;
        return (y == Integer.MAX_VALUE) ? 0 : y;
    }

    /** Walks the polynomial, finds supported y-runs in the BEV mask, draws each as an arc. */
    private void drawValidatedSegments(Mat output, Mat bev, double[] coef,
                                       int yMin, int yMax, Scalar color) {
        List<int[]> intervals = findSupportedIntervals(bev, coef, yMin, yMax);
        if (intervals.isEmpty()) return;

        int step = Math.max(1, config.drawStepPixels);
        for (int[] interval : intervals) {
            int yA = interval[0];
            int yB = interval[1];
            List<Point> bevSamples = new ArrayList<>();
            for (int y = yA; y <= yB; y += step) {
                bevSamples.add(new Point(evalPoly(coef, y), y));
            }
            // Always include the very last y of the interval so the segment reaches the
            // end of the supported run exactly.
            if (bevSamples.isEmpty() || bevSamples.get(bevSamples.size() - 1).y < yB) {
                bevSamples.add(new Point(evalPoly(coef, yB), yB));
            }
            if (bevSamples.size() < 2) continue;

            MatOfPoint2f src2f = new MatOfPoint2f(bevSamples.toArray(new Point[0]));
            MatOfPoint2f dst2f = new MatOfPoint2f();
            try {
                Core.perspectiveTransform(src2f, dst2f, bevToOrigMatrix);
                drawPolyline(output, dst2f.toArray(), color);
            } finally {
                src2f.release();
                dst2f.release();
            }
        }
    }

    /**
     * For each row y in [yMin, yMax], check whether the BEV mask has any non-zero pixel
     * within ±{@code outputSupportHalfWidthPx} of {@code poly(y)}. Group consecutive
     * supported rows into {@code [yStart, yEnd]} intervals; drop intervals shorter than
     * {@code outputMinRunLength} so isolated stray pixels are not promoted to dashes.
     */
    private List<int[]> findSupportedIntervals(Mat bev, double[] coef, int yMin, int yMax) {
        int halfW    = config.outputSupportHalfWidthPx;
        int minRun   = config.outputMinRunLength;
        int W        = bev.cols();

        List<int[]> intervals = new ArrayList<>();
        int runStart = -1;

        for (int y = yMin; y <= yMax; y++) {
            int xCenter = (int) Math.round(evalPoly(coef, y));
            int xLow    = Math.max(0, xCenter - halfW);
            int xHigh   = Math.min(W, xCenter + halfW + 1);
            boolean supported = false;
            if (xLow < xHigh) {
                Mat row = bev.submat(y, y + 1, xLow, xHigh);
                try { supported = Core.countNonZero(row) > 0; }
                finally { row.release(); }
            }
            if (supported) {
                if (runStart < 0) runStart = y;
            } else if (runStart >= 0) {
                if (y - runStart >= minRun) intervals.add(new int[]{runStart, y - 1});
                runStart = -1;
            }
        }
        if (runStart >= 0 && yMax - runStart + 1 >= minRun) {
            intervals.add(new int[]{runStart, yMax});
        }
        return intervals;
    }

    /**
     * Test a hypothesised polynomial against the BEV mask. Accept if at least
     * {@code hypothesisMinSupportFraction} of the y-range has paint near the curve. Cheap
     * sanity check: walks every {@code drawStepPixels} rows.
     */
    private boolean hypothesisIsSupported(Mat bev, double[] coef) {
        int yMax    = config.bevHeight - 1;
        int halfW   = config.outputSupportHalfWidthPx;
        int step    = Math.max(1, config.drawStepPixels);
        int W       = bev.cols();
        int total   = 0;
        int hits    = 0;
        for (int y = 0; y <= yMax; y += step) {
            int xCenter = (int) Math.round(evalPoly(coef, y));
            int xLow    = Math.max(0, xCenter - halfW);
            int xHigh   = Math.min(W, xCenter + halfW + 1);
            if (xLow >= xHigh) continue;
            total++;
            Mat row = bev.submat(y, y + 1, xLow, xHigh);
            try { if (Core.countNonZero(row) > 0) hits++; }
            finally { row.release(); }
        }
        if (total == 0) return false;
        return (double) hits / total >= config.hypothesisMinSupportFraction;
    }

    /** Parallel-shift a polynomial along x: add {@code offsetX} to its constant term. */
    private static double[] shiftPolynomial(double[] coef, double offsetX) {
        double[] shifted = coef.clone();
        shifted[shifted.length - 1] += offsetX;     // last term is the constant
        return shifted;
    }

    /**
     * Update the EMA of BEV-pixel lane width using the bottom of the two polynomials, where
     * the perspective projection is the most accurate and lane width is least distorted by
     * curvature. Sanity-rejects implausible widths (negative, or wider than 80% of BEV).
     */
    private void updateLearnedLaneWidth(double[] leftCoef, double[] rightCoef) {
        if (leftCoef == null || rightCoef == null) return;
        int yMeas = config.bevHeight - 1;
        double left  = evalPoly(leftCoef,  yMeas);
        double right = evalPoly(rightCoef, yMeas);
        double w     = right - left;
        if (w <= 0 || w > config.bevWidth * 0.8) return;
        if (Double.isNaN(learnedLaneWidthBev)) learnedLaneWidthBev = w;
        else {
            double a = config.laneWidthAlpha;
            learnedLaneWidthBev = (1 - a) * learnedLaneWidthBev + a * w;
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

    // =========================================================================================
    // Polynomial least-squares fit (x = c0·y^d + c1·y^(d-1) + … + cd)
    // =========================================================================================

    /**
     * Fit a polynomial robustly: do an initial least-squares fit, then iteratively drop pixels
     * whose horizontal residual exceeds {@code residualThresholdPx} and refit. Wheels, cars,
     * shadow edges and other off-line clusters that briefly enter the sliding window are
     * stripped out, so the final polynomial traces the true lane line tightly.
     *
     * <p>Mutates the provided {@code points} list so that on return it holds only the inliers
     * used for the final fit. This keeps drawing consistent with the fit (the drawn curve
     * spans the inlier y-range).
     */
    private double[] robustFit(List<Point> points) {
        if (points.size() < config.minPointsForFit) return null;
        double[] coef = fitPolynomial(points, config.polynomialDegree);
        if (coef == null) return null;

        double threshold = config.residualThresholdPx;
        for (int it = 0; it < config.refinementIterations; it++) {
            List<Point> inliers = new ArrayList<>(points.size());
            for (Point p : points) {
                if (Math.abs(p.x - evalPoly(coef, p.y)) <= threshold) inliers.add(p);
            }
            // If outlier rejection ate too much, keep the previous fit — better than nothing.
            if (inliers.size() < config.minPointsForFit) break;
            if (inliers.size() == points.size()) break;            // converged
            double[] refit = fitPolynomial(inliers, config.polynomialDegree);
            if (refit == null) break;
            coef = refit;
            points.clear();
            points.addAll(inliers);
        }
        return coef;
    }

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
