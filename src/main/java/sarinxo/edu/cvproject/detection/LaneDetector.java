package sarinxo.edu.cvproject.detection;


import org.opencv.core.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.*;

/**
 * Production-ready lane detector based on classical computer vision (OpenCV).
 * For dashcam video, no neural networks.
 *
 * <h3>Pipeline (per frame)</h3>
 * <ol>
 *   <li>Bilateral filter — denoise while preserving edges.</li>
 *   <li>CLAHE on L-channel (HLS) — handles dark→light transitions like tunnels.</li>
 *   <li>Color thresholding (white ∪ dirt-tolerant yellow).</li>
 *   <li>Morphological denoising (opening removes specks, closing joins dashes).</li>
 *   <li>Inverse Perspective Mapping (IPM) into bird's-eye view.</li>
 *   <li><b>Shape validation</b> — analyze each connected component in bird's-eye:
 *       <ul>
 *         <li>length in valid range (rejects specks AND whole car bodies),</li>
 *         <li>width below max (rejects wheels, white car parts, signs),</li>
 *         <li>aspect ratio high (rejects chunky shapes),</li>
 *         <li>orientation near-vertical (rejects horizontal stuff like crosswalks
 *             from a perpendicular angle, shadows, road edges of buildings).</li>
 *       </ul>
 *       Only validated components survive into the lane mask.
 *       If too few survive, the frame is declared marking-less and nothing is drawn.</li>
 *   <li>Center-biased histogram peak detection on the validated mask.</li>
 *   <li>Sliding-window pixel collection.</li>
 *   <li>Cubic polynomial fit (x = a·y³ + b·y² + c·y + d).</li>
 *   <li>3-state machine per side: LOST → ACQUIRING → TRACKED. Drawing only in TRACKED.</li>
 *   <li>Sample N points along the cubic in bird's-eye, transform points
 *       (not the image) through inverse homography → draw a polyline of N points
 *       in the original frame. Curvature is preserved.</li>
 * </ol>
 *
 * <p><b>Stateful, NOT thread-safe.</b> One instance per video stream.
 */
public class LaneDetector {

    // ===============================================================================================
    //  Configuration
    // ===============================================================================================

    public static class Config {
        // --- Preprocessing ---
        public int    bilateralDiameter   = 7;
        public double bilateralSigmaColor = 50.0;
        public double bilateralSigmaSpace = 50.0;
        public double claheClipLimit      = 2.0;
        public Size   claheTileSize       = new Size(8, 8);

        // --- Color thresholds (HLS — H 0..179, L 0..255, S 0..255) ---
        public Scalar whiteLower  = new Scalar(  0, 200,   0);
        public Scalar whiteUpper  = new Scalar(179, 255, 255);
        public Scalar yellowLower = new Scalar( 15,  60,  60);
        public Scalar yellowUpper = new Scalar( 40, 255, 255);

        // --- Morphology on color mask ---
        public Size morphOpenKernel  = new Size(2, 2);   // remove tiny specks
        public Size morphCloseKernel = new Size(3, 9);   // join dashed segments vertically

        // --- ROI / IPM trapezoid (proportions of frame size) ---
        public double roiTopY          = 0.62;
        public double roiBottomY       = 0.95;
        public double roiTopLeftX      = 0.42;
        public double roiTopRightX     = 0.58;
        public double roiBottomLeftX   = 0.08;
        public double roiBottomRightX  = 0.92;
        public double warpDestLeftX    = 0.25;
        public double warpDestRightX   = 0.75;

        // --- Shape validation (in bird's-eye view) ---
        // Sizes are RATIOS of bird's-eye dimensions, so they auto-scale with resolution.
        // Width is measured along the rotated bbox's short side, length along the long side.
        public double minMarkingLengthRatio = 0.04;  // shorter than 4% of height → speck or noise
        public double maxMarkingLengthRatio = 0.95;  // longer than 95% → spans almost whole image (rare, OK)
        public double minMarkingWidthRatio  = 0.001; // sanity floor
        public double maxMarkingWidthRatio  = 0.05;  // wider than 5% of width → car body / wheel / sign
        public double minAspectRatio        = 4.0;   // length / width must be ≥ this
        public double maxOrientationDegrees = 35.0;  // angle from vertical in bird's-eye; >35° = transverse
        public double minExtent             = 0.30;  // area / rotatedBbox area; very low = sparse junk
        public double maxExtent             = 0.95;  // very high + chunky shape = blob, not marking
        public int    minValidComponents    = 1;     // need at least this many to consider frame "has markings"

        // --- Center-biased peak search ---
        public double centerSearchRatio = 0.40;

        // --- Sliding window ---
        public int nWindows  = 9;
        public int marginPx  = 80;
        public int minPix    = 50;

        // --- Polynomial fit ---
        public int polynomialOrder = 3;

        // --- Detection-quality thresholds (mandatory for state-machine "good" frame) ---
        public int    minPointsForFit  = 30;
        public double minYCoverage     = 0.35;

        // --- State machine ---
        public int historySize       = 4;
        public int framesToAcquire   = 3;
        public int gracePeriodFrames = 3;

        // --- Drawing ---
        public Scalar  leftLaneColor  = new Scalar(  0, 255, 255);   // yellow (BGR)
        public Scalar  rightLaneColor = new Scalar(  0, 255, 255);
        public Scalar  laneFillColor  = new Scalar(  0, 200,   0);
        public int     lineThickness  = 10;
        public int     curveSamples   = 60;
        public double  overlayAlpha   = 0.5;
        public boolean fillLane       = true;
    }

    private final Config config;

    // ===============================================================================================
    //  State
    // ===============================================================================================

    private enum LaneState { LOST, ACQUIRING, TRACKED }

    private static class SideState {
        final Deque<double[]> history = new ArrayDeque<>();
        LaneState state      = LaneState.LOST;
        int       goodStreak = 0;
        int       badStreak  = 0;
    }

    private final SideState left  = new SideState();
    private final SideState right = new SideState();
    private long    frameCounter             = 0;
    private boolean lastFrameHadValidShapes  = false; // diagnostics

    // Lazy-initialized per resolution
    private Size  frameSize;
    private Size  warpedSize;
    private Mat   perspectiveMatrix;
    private Mat   inversePerspectiveMatrix;
    private Mat   morphOpenK;
    private Mat   morphCloseK;
    private CLAHE clahe;

    // Pre-computed shape-validation thresholds (in pixels — recomputed on resize)
    private double minLenPx, maxLenPx, minWidPx, maxWidPx;

    // ===============================================================================================
    //  Public API
    // ===============================================================================================

    public LaneDetector() { this(new Config()); }

    public LaneDetector(Config config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    /**
     * Processes one frame and returns an annotated copy.
     * Caller owns the returned Mat and must release it.
     */
    public Mat processFrame(Mat frame) {
        if (frame == null || frame.empty()) {
            throw new IllegalArgumentException("Input frame is null or empty");
        }
        if (frame.channels() != 3) {
            throw new IllegalArgumentException(
                    "Expected 3-channel BGR frame, got " + frame.channels());
        }
        if (frameSize == null
                || frame.width()  != frameSize.width
                || frame.height() != frameSize.height) {
            initializeForSize(frame.size());
        }
        frameCounter++;

        Mat preprocessed   = null;
        Mat colorMask      = null;
        Mat warpedColor    = null;
        Mat validatedMask  = null;
        try {
            preprocessed = preprocess(frame);
            colorMask    = createColorMask(preprocessed);
            warpedColor  = warpToBirdsEye(colorMask);

            // SHAPE VALIDATION: only keep connected components that look like lane markings
            ShapeValidationResult sv = validateShapes(warpedColor);
            validatedMask              = sv.mask;
            lastFrameHadValidShapes    = sv.validCount >= config.minValidComponents;

            // If no valid markings present in this frame, force "bad" frame for both sides.
            // The state machine will keep TRACKED for gracePeriod frames; if the road truly
            // has no markings, drawing stops cleanly within gracePeriod.
            LanePixels px;
            if (lastFrameHadValidShapes) {
                px = findLanePixels(validatedMask);
            } else {
                px = new LanePixels(); // empty → both sides see no pixels → bad frame
            }

            updateSide(left,  px.leftX,  px.leftY,  validatedMask.rows());
            updateSide(right, px.rightX, px.rightY, validatedMask.rows());

            double[] leftFit  = (left.state  == LaneState.TRACKED) ? averageFits(left.history)  : null;
            double[] rightFit = (right.state == LaneState.TRACKED) ? averageFits(right.history) : null;

            return drawLanes(frame, leftFit, rightFit);
        } finally {
            if (preprocessed  != null) preprocessed.release();
            if (colorMask     != null) colorMask.release();
            if (warpedColor   != null) warpedColor.release();
            if (validatedMask != null) validatedMask.release();
        }
    }

    /** Diagnostic snapshot. */
    public DebugInfo getDebugInfo() {
        return new DebugInfo(
                frameCounter, lastFrameHadValidShapes,
                left.state.name(),  left.goodStreak,  left.badStreak,  left.history.size(),
                right.state.name(), right.goodStreak, right.badStreak, right.history.size());
    }

    public static class DebugInfo {
        public final long    frameCount;
        public final boolean hadValidShapes;
        public final String  leftState,  rightState;
        public final int     leftGood,   rightGood;
        public final int     leftBad,    rightBad;
        public final int     leftHist,   rightHist;
        DebugInfo(long fc, boolean hvs,
                  String ls, int lg, int lb, int lh,
                  String rs, int rg, int rb, int rh) {
            this.frameCount = fc; this.hadValidShapes = hvs;
            this.leftState  = ls; this.leftGood  = lg; this.leftBad  = lb; this.leftHist  = lh;
            this.rightState = rs; this.rightGood = rg; this.rightBad = rb; this.rightHist = rh;
        }
    }

    public void reset() {
        resetSide(left);
        resetSide(right);
        frameCounter = 0;
        lastFrameHadValidShapes = false;
    }

    private static void resetSide(SideState s) {
        s.history.clear();
        s.state      = LaneState.LOST;
        s.goodStreak = 0;
        s.badStreak  = 0;
    }

    public void release() {
        reset();
        if (perspectiveMatrix        != null) { perspectiveMatrix.release();        perspectiveMatrix        = null; }
        if (inversePerspectiveMatrix != null) { inversePerspectiveMatrix.release(); inversePerspectiveMatrix = null; }
        if (morphOpenK               != null) { morphOpenK.release();               morphOpenK               = null; }
        if (morphCloseK              != null) { morphCloseK.release();              morphCloseK              = null; }
        frameSize  = null;
        warpedSize = null;
        clahe      = null;
    }

    // ===============================================================================================
    //  Initialization
    // ===============================================================================================

    private void initializeForSize(Size size) {
        if (perspectiveMatrix        != null) perspectiveMatrix.release();
        if (inversePerspectiveMatrix != null) inversePerspectiveMatrix.release();
        if (morphOpenK               != null) morphOpenK.release();
        if (morphCloseK              != null) morphCloseK.release();

        this.frameSize  = size;
        this.warpedSize = size;

        double w = size.width, h = size.height;

        MatOfPoint2f src = new MatOfPoint2f(
                new Point(config.roiTopLeftX     * w, config.roiTopY    * h),
                new Point(config.roiTopRightX    * w, config.roiTopY    * h),
                new Point(config.roiBottomRightX * w, config.roiBottomY * h),
                new Point(config.roiBottomLeftX  * w, config.roiBottomY * h));
        MatOfPoint2f dst = new MatOfPoint2f(
                new Point(config.warpDestLeftX  * w, 0),
                new Point(config.warpDestRightX * w, 0),
                new Point(config.warpDestRightX * w, h),
                new Point(config.warpDestLeftX  * w, h));

        perspectiveMatrix        = Imgproc.getPerspectiveTransform(src, dst);
        inversePerspectiveMatrix = Imgproc.getPerspectiveTransform(dst, src);
        src.release(); dst.release();

        morphOpenK  = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, config.morphOpenKernel);
        morphCloseK = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, config.morphCloseKernel);

        clahe = Imgproc.createCLAHE(config.claheClipLimit, config.claheTileSize);

        // pre-compute shape-validation thresholds in pixels
        minLenPx = h * config.minMarkingLengthRatio;
        maxLenPx = h * config.maxMarkingLengthRatio;
        minWidPx = w * config.minMarkingWidthRatio;
        maxWidPx = w * config.maxMarkingWidthRatio;

        reset();
    }

    // ===============================================================================================
    //  Stage 1: preprocess
    // ===============================================================================================

    private Mat preprocess(Mat frame) {
        Mat denoised = new Mat();
        Imgproc.bilateralFilter(frame, denoised,
                config.bilateralDiameter, config.bilateralSigmaColor, config.bilateralSigmaSpace);

        Mat hls = new Mat();
        Imgproc.cvtColor(denoised, hls, Imgproc.COLOR_BGR2HLS);
        denoised.release();

        List<Mat> ch = new ArrayList<>(3);
        Core.split(hls, ch);
        Mat lEq = new Mat();
        clahe.apply(ch.get(1), lEq);
        ch.get(1).release();
        ch.set(1, lEq);
        Core.merge(ch, hls);
        for (Mat c : ch) c.release();

        return hls;
    }

    // ===============================================================================================
    //  Stage 2: color mask + morphological denoising
    // ===============================================================================================

    private Mat createColorMask(Mat hls) {
        Mat whiteMask  = new Mat();
        Mat yellowMask = new Mat();
        Core.inRange(hls, config.whiteLower,  config.whiteUpper,  whiteMask);
        Core.inRange(hls, config.yellowLower, config.yellowUpper, yellowMask);

        Mat colorMask = new Mat();
        Core.bitwise_or(whiteMask, yellowMask, colorMask);
        whiteMask.release();
        yellowMask.release();

        // Opening — kill specks/noise smaller than the kernel
        Imgproc.morphologyEx(colorMask, colorMask, Imgproc.MORPH_OPEN, morphOpenK);
        // Closing — bridge dashed lane segments into single connected components
        Imgproc.morphologyEx(colorMask, colorMask, Imgproc.MORPH_CLOSE, morphCloseK);

        return colorMask;
    }

    // ===============================================================================================
    //  Stage 3: bird's-eye warp
    // ===============================================================================================

    private Mat warpToBirdsEye(Mat binary) {
        Mat warped = new Mat();
        Imgproc.warpPerspective(binary, warped, perspectiveMatrix, warpedSize,
                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, new Scalar(0));
        return warped;
    }

    // ===============================================================================================
    //  Stage 4: SHAPE VALIDATION — keep only marking-shaped components
    // ===============================================================================================

    private static class ShapeValidationResult {
        final Mat mask;
        final int validCount;
        ShapeValidationResult(Mat mask, int validCount) {
            this.mask = mask;
            this.validCount = validCount;
        }
    }

    /**
     * Iterates over every connected component in the (binary) bird's-eye color mask
     * and keeps only those whose <b>shape</b> matches a lane marking:
     * <ul>
     *   <li>length in [minLenPx, maxLenPx],</li>
     *   <li>width  in [minWidPx, maxWidPx],</li>
     *   <li>aspect ratio (length/width) ≥ minAspectRatio,</li>
     *   <li>orientation within ±maxOrientationDegrees of vertical,</li>
     *   <li>extent (area/rotatedBboxArea) in [minExtent, maxExtent].</li>
     * </ul>
     *
     * <p>Performed in bird's-eye view because lane markings have stable geometric
     * properties there (width/length don't change with distance), so a single set
     * of thresholds works at any depth in the scene.
     *
     * <p>Each surviving component is drawn into the output mask via {@code findContours}
     * + {@code drawContours} so the resulting mask contains <b>only</b> validated pixels.
     */
    private ShapeValidationResult validateShapes(Mat colorMaskBev) {
        // Find external contours — ideal for connected components in a binary image.
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(colorMaskBev, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();

        Mat outMask = Mat.zeros(colorMaskBev.size(), CvType.CV_8UC1);
        List<MatOfPoint> kept = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            if (isMarkingShape(contour)) {
                kept.add(contour);
            }
        }

        if (!kept.isEmpty()) {
            Imgproc.drawContours(outMask, kept, -1, new Scalar(255), Core.FILLED);
        }

        for (MatOfPoint c : contours) c.release();
        return new ShapeValidationResult(outMask, kept.size());
    }

    private boolean isMarkingShape(MatOfPoint contour) {
        // Need at least 5 points for minAreaRect to be meaningful.
        if (contour.total() < 5) return false;

        double area = Imgproc.contourArea(contour);
        if (area < 1.0) return false;

        MatOfPoint2f c2f = new MatOfPoint2f(contour.toArray());
        RotatedRect rr = Imgproc.minAreaRect(c2f);
        c2f.release();

        double w = rr.size.width;
        double h = rr.size.height;
        double length = Math.max(w, h);
        double width  = Math.min(w, h);

        if (width  < 1e-3)             return false;
        if (length < minLenPx)         return false;   // too short → noise speck
        if (length > maxLenPx)         return false;   // unusually long → reject
        if (width  < minWidPx)         return false;   // sub-pixel sliver
        if (width  > maxWidPx)         return false;   // too wide → wheel, car body, sign

        double aspect = length / width;
        if (aspect < config.minAspectRatio) return false; // chunky → not a marking

        // Orientation: angle of the LONG side from vertical (in bird's-eye, markings should
        // run nearly vertical; transverse stuff like crosswalk bars or cracks is rejected).
        double angleFromVertical = orientationFromVertical(rr);
        if (angleFromVertical > config.maxOrientationDegrees) return false;

        // Extent: how completely does the contour fill its rotated bounding box?
        double rotatedBboxArea = w * h;
        if (rotatedBboxArea < 1.0) return false;
        double extent = area / rotatedBboxArea;
        if (extent < config.minExtent || extent > config.maxExtent) return false;

        return true;
    }

    /**
     * Returns the absolute angular distance (degrees, 0..90) between the rectangle's
     * long axis and the vertical image axis. OpenCV's RotatedRect.angle is the
     * rotation of the FIRST side from horizontal — we have to interpret it carefully.
     */
    private static double orientationFromVertical(RotatedRect rr) {
        double angleDeg = rr.angle;            // OpenCV: angle of width-side from horizontal, range (-90, 0]
        double w = rr.size.width;
        double h = rr.size.height;
        // If width > height, the long axis IS the "width" side at angleDeg from horizontal.
        // If width < height, the long axis is the "height" side, perpendicular → +90°.
        double longAxisFromHorizontal = (w >= h) ? angleDeg : angleDeg + 90.0;
        // Distance to vertical (90° from horizontal), modulo 180°
        double diff = Math.abs(longAxisFromHorizontal - 90.0);
        diff = diff % 180.0;
        if (diff > 90.0) diff = 180.0 - diff;
        return diff;
    }

    // ===============================================================================================
    //  Stage 5: locate lane pixels (center-biased) + sliding windows
    // ===============================================================================================

    private static class LanePixels {
        final List<Double> leftX  = new ArrayList<>();
        final List<Double> leftY  = new ArrayList<>();
        final List<Double> rightX = new ArrayList<>();
        final List<Double> rightY = new ArrayList<>();
    }

    private LanePixels findLanePixels(Mat warped) {
        LanePixels out = new LanePixels();
        int height = warped.rows();
        int width  = warped.cols();
        if (height == 0 || width == 0) return out;

        Mat bottom    = warped.submat(height / 2, height, 0, width);
        Mat histogram = new Mat();
        Core.reduce(bottom, histogram, 0, Core.REDUCE_SUM, CvType.CV_32S);
        bottom.release();

        int midpoint     = width / 2;
        int searchOffset = (int) (width * config.centerSearchRatio);
        int leftLow      = Math.max(0,     midpoint - searchOffset);
        int rightHigh    = Math.min(width, midpoint + searchOffset);

        int leftBase  = peakColumn(histogram, leftLow,  midpoint,  /*preferRightmost*/ true);
        int rightBase = peakColumn(histogram, midpoint, rightHigh, /*preferRightmost*/ false);
        histogram.release();

        Mat nonZero = new Mat();
        Core.findNonZero(warped, nonZero);
        int total = (int) nonZero.total();
        if (total == 0) {
            nonZero.release();
            return out;
        }
        int[] data = new int[total * 2];
        nonZero.get(0, 0, data);
        nonZero.release();

        int windowHeight = height / config.nWindows;
        int leftCurrent  = leftBase;
        int rightCurrent = rightBase;

        for (int w = 0; w < config.nWindows; w++) {
            int yLow  = height - (w + 1) * windowHeight;
            int yHigh = height -  w      * windowHeight;
            int xLL   = leftCurrent  - config.marginPx;
            int xLH   = leftCurrent  + config.marginPx;
            int xRL   = rightCurrent - config.marginPx;
            int xRH   = rightCurrent + config.marginPx;

            int leftSum = 0, leftCount = 0, rightSum = 0, rightCount = 0;

            for (int i = 0; i < total; i++) {
                int x = data[i * 2];
                int y = data[i * 2 + 1];
                if (y < yLow || y >= yHigh) continue;
                if (x >= xLL && x < xLH) {
                    out.leftX.add((double) x);
                    out.leftY.add((double) y);
                    leftSum += x; leftCount++;
                } else if (x >= xRL && x < xRH) {
                    out.rightX.add((double) x);
                    out.rightY.add((double) y);
                    rightSum += x; rightCount++;
                }
            }
            if (leftCount  > config.minPix) leftCurrent  = leftSum  / leftCount;
            if (rightCount > config.minPix) rightCurrent = rightSum / rightCount;
        }
        return out;
    }

    private static int peakColumn(Mat histogram1xN, int startCol, int endCol, boolean preferRightmost) {
        int  peakCol = startCol;
        long peakVal = -1L;
        if (preferRightmost) {
            for (int c = startCol; c < endCol; c++) {
                long v = (long) histogram1xN.get(0, c)[0];
                if (v >= peakVal) { peakVal = v; peakCol = c; }
            }
        } else {
            for (int c = endCol - 1; c >= startCol; c--) {
                long v = (long) histogram1xN.get(0, c)[0];
                if (v >= peakVal) { peakVal = v; peakCol = c; }
            }
        }
        return peakCol;
    }

    // ===============================================================================================
    //  Stage 6: state machine + polynomial fit
    // ===============================================================================================

    private void updateSide(SideState s, List<Double> xs, List<Double> ys, int height) {
        boolean qualifies = qualifiesForFit(ys, height);
        double[] freshFit = qualifies ? fitPolynomial(xs, ys, config.polynomialOrder) : null;

        if (freshFit != null) {
            s.history.addLast(freshFit);
            while (s.history.size() > config.historySize) s.history.removeFirst();
            s.goodStreak++;
            s.badStreak = 0;
            if (s.state == LaneState.LOST || s.state == LaneState.ACQUIRING) {
                s.state = (s.goodStreak >= config.framesToAcquire)
                        ? LaneState.TRACKED
                        : LaneState.ACQUIRING;
            }
        } else {
            s.badStreak++;
            s.goodStreak = 0;
            if (s.state == LaneState.TRACKED) {
                if (s.badStreak >= config.gracePeriodFrames) {
                    s.state = LaneState.LOST;
                    s.history.clear();
                }
                // else: still TRACKED but no new fit added — smoothing window will
                // shrink as old frames age out of the deque (we keep up to historySize).
            } else {
                // ACQUIRING / LOST: any miss → reset
                s.state = LaneState.LOST;
                s.history.clear();
            }
        }
    }

    private boolean qualifiesForFit(List<Double> ys, int height) {
        int n = ys.size();
        if (n < config.minPointsForFit || height <= 0) return false;
        double minY = Double.POSITIVE_INFINITY, maxY = Double.NEGATIVE_INFINITY;
        for (double y : ys) {
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        return (maxY - minY) / height >= config.minYCoverage;
    }

    private static double[] fitPolynomial(List<Double> xs, List<Double> ys, int order) {
        int n = xs.size();
        if (n < order + 3) return null;

        Mat A = new Mat(n, order + 1, CvType.CV_64F);
        Mat b = new Mat(n, 1,         CvType.CV_64F);
        for (int i = 0; i < n; i++) {
            double y = ys.get(i);
            double pow = 1.0;
            for (int j = order; j >= 0; j--) {
                A.put(i, j, pow);
                pow *= y;
            }
            b.put(i, 0, xs.get(i));
        }

        Mat coeffs = new Mat();
        boolean ok = Core.solve(A, b, coeffs, Core.DECOMP_QR);
        A.release(); b.release();
        if (!ok) { coeffs.release(); return null; }

        double[] result = new double[order + 1];
        for (int i = 0; i <= order; i++) result[i] = coeffs.get(i, 0)[0];
        coeffs.release();

        for (double v : result) if (Double.isNaN(v) || Double.isInfinite(v)) return null;
        return result;
    }

    private static double[] averageFits(Deque<double[]> history) {
        if (history.isEmpty()) return null;
        int order = history.peekLast().length;
        double[] sum = new double[order];
        for (double[] f : history) for (int i = 0; i < order; i++) sum[i] += f[i];
        int n = history.size();
        for (int i = 0; i < order; i++) sum[i] /= n;
        return sum;
    }

    private static double evaluatePoly(double[] coeffs, double y) {
        double v = coeffs[0];
        for (int i = 1; i < coeffs.length; i++) v = v * y + coeffs[i];
        return v;
    }

    // ===============================================================================================
    //  Stage 7: render — sample many points, transform points (not image), draw curve
    // ===============================================================================================

    private List<Point> sampleCurveInOriginalFrame(double[] fit) {
        int height = (int) warpedSize.height;
        int width  = (int) warpedSize.width;

        int samples = Math.max(8, config.curveSamples);
        List<Point> bevPts = new ArrayList<>(samples);
        double step = (height - 1.0) / (samples - 1);
        for (int i = 0; i < samples; i++) {
            double y = i * step;
            double x = evaluatePoly(fit, y);
            if (x < 0 || x >= width) continue;
            bevPts.add(new Point(x, y));
        }
        if (bevPts.size() < 2) return Collections.emptyList();

        MatOfPoint2f srcPts = new MatOfPoint2f();
        srcPts.fromList(bevPts);
        MatOfPoint2f dstPts = new MatOfPoint2f();
        Core.perspectiveTransform(srcPts, dstPts, inversePerspectiveMatrix);
        List<Point> originalPts = dstPts.toList();
        srcPts.release();
        dstPts.release();

        List<Point> clipped = new ArrayList<>(originalPts.size());
        double fw = frameSize.width, fh = frameSize.height;
        for (Point p : originalPts) {
            if (p.x >= -10 && p.x <= fw + 10 && p.y >= -10 && p.y <= fh + 10) {
                clipped.add(p);
            }
        }
        return clipped;
    }

    private Mat drawLanes(Mat originalFrame, double[] leftFit, double[] rightFit) {
        Mat overlay = Mat.zeros(frameSize, CvType.CV_8UC3);

        List<Point> leftCurve  = (leftFit  != null) ? sampleCurveInOriginalFrame(leftFit)  : Collections.emptyList();
        List<Point> rightCurve = (rightFit != null) ? sampleCurveInOriginalFrame(rightFit) : Collections.emptyList();

        if (config.fillLane && leftCurve.size() >= 2 && rightCurve.size() >= 2) {
            List<Point> poly = new ArrayList<>(leftCurve.size() + rightCurve.size());
            for (int i = leftCurve.size() - 1; i >= 0; i--) poly.add(leftCurve.get(i));
            poly.addAll(rightCurve);
            MatOfPoint mp = new MatOfPoint();
            mp.fromList(poly);
            Imgproc.fillPoly(overlay, Collections.singletonList(mp), config.laneFillColor);
            mp.release();
        }

        if (leftCurve.size()  >= 2) drawPolyline(overlay, leftCurve,  config.leftLaneColor);
        if (rightCurve.size() >= 2) drawPolyline(overlay, rightCurve, config.rightLaneColor);

        Mat result = new Mat();
        Core.addWeighted(originalFrame, 1.0, overlay, config.overlayAlpha, 0, result);
        overlay.release();
        return result;
    }

    private void drawPolyline(Mat dst, List<Point> pts, Scalar color) {
        MatOfPoint mp = new MatOfPoint();
        mp.fromList(pts);
        Imgproc.polylines(dst, Collections.singletonList(mp),
                /*isClosed*/ false, color, config.lineThickness, Imgproc.LINE_AA, 0);
        mp.release();
    }
}