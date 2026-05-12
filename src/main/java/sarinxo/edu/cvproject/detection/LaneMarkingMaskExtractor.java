package sarinxo.edu.cvproject.detection;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;

/**
 * Extracts a binary mask of road lane markings (white and yellow) from a single video frame.
 *
 * <h2>Precision strategy</h2>
 * The pipeline is precision-first: a pixel is kept only if <b>two independent pieces of
 * evidence agree</b> on it.
 * <ul>
 *   <li><b>Colour evidence</b> — the pixel looks like lane paint (yellow in LAB B-channel or
 *       white in HLS / high-brightness grayscale).</li>
 *   <li><b>Structural evidence</b> — the pixel is part of a thin bright structure, isolated
 *       by a Top-Hat morphological transform.</li>
 * </ul>
 * Their <b>intersection</b> (AND) rejects most false positives that plague colour-only masks:
 * white cars (bright but not thin), concrete (right colour but no local peak), sky reflections,
 * dried grass, etc.
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Gaussian blur — suppresses sensor noise on low-quality footage.</li>
 *   <li>CLAHE on the L-channel of LAB — normalises local contrast so the mask survives
 *       sudden illumination changes (e.g. exiting a tunnel).</li>
 *   <li>Colour mask = yellow (LAB B-channel + L brightness gate) ∪ white (HLS + grayscale).</li>
 *   <li>Structural mask = threshold(Top-Hat).</li>
 *   <li>Fusion = colour AND structural in {@link FusionMode#STRICT} (default); OR in
 *       {@link FusionMode#PERMISSIVE}.</li>
 *   <li>Light morphological opening with a 3×3 ellipse — removes salt noise without thickening
 *       lines. Closing is disabled by default; dashed-line reconnection belongs in a later
 *       stage (sliding window in BEV, Hough {@code maxLineGap}, Kalman).</li>
 *   <li>Connected-components area filter — drops blobs below a relative area threshold.</li>
 *   <li>Optional skeletonization to single-pixel-wide centerlines (iterative-erosion algorithm).</li>
 *   <li>Temporal agreement filter — keep pixels seen in at least <i>K</i> of the last
 *       <i>N</i> frames, eliminating flicker.</li>
 * </ol>
 *
 * <h2>Recall vs. precision trade-off</h2>
 * If the default mask is too sparse on very faded paint, switch to
 * {@link FusionMode#PERMISSIVE} or loosen colour thresholds via {@link Config.Builder}.
 *
 * <h2>API</h2>
 * <pre>{@code
 * LaneMarkingMaskExtractor extractor = new LaneMarkingMaskExtractor(
 *         Config.builder().skeletonize(true).build());
 * try {
 *     while (videoHasNextFrame()) {
 *         Mat frame = readFrame();          // CV_8UC3, BGR
 *         Mat mask  = extractor.process(frame);
 *         try {
 *             // feed mask into the next stage of the pipeline
 *         } finally { mask.release(); }
 *         frame.release();
 *     }
 * } finally { extractor.release(); }
 * }</pre>
 *
 * <h2>Lifecycle &amp; threading</h2>
 * The returned mask is a freshly allocated {@code Mat} owned by the caller — release it.
 * Call {@link #reset()} between unrelated videos. Call {@link #release()} when finished.
 * Instances are <b>not</b> thread-safe.
 */
public final class LaneMarkingMaskExtractor {

    /** How colour evidence and structural evidence are combined. */
    public enum FusionMode {
        /** Intersection (AND) — high precision, default. */
        STRICT,
        /** Union (OR) — high recall, useful for very faded paint. */
        PERMISSIVE
    }

    // =========================================================================================
    // Configuration
    // =========================================================================================

    /** Immutable bundle of tuneable parameters. Build via {@link Config#builder()}. */
    public static final class Config {

        // -- Preprocessing --
        public final int    gaussianKernelSize;
        public final double claheClipLimit;
        public final Size   claheTileGridSize;

        // -- Yellow (LAB B-channel) --
        public final int labBMin;
        public final int labBMax;
        public final int labLMinForYellow;

        // -- White (HLS + grayscale) --
        public final int hlsLMin;
        public final int hlsSMax;
        public final int grayWhiteMin;

        // -- Top-Hat structural cue --
        public final Size topHatKernelSize;
        public final int  topHatThreshold;

        // -- Fusion --
        public final FusionMode fusionMode;

        // -- Cleanup --
        public final Size    openingKernelSize;
        public final boolean closingEnabled;
        public final Size    closingKernelSize;

        // -- Connected-component filtering --
        public final double minComponentAreaRatio;

        // -- Skeletonization --
        public final boolean skeletonize;
        public final int     skeletonMaxIterations;

        // -- Temporal smoothing --
        public final int temporalWindowSize;
        public final int temporalMinAgreement;

        private Config(Builder b) {
            this.gaussianKernelSize    = b.gaussianKernelSize;
            this.claheClipLimit        = b.claheClipLimit;
            this.claheTileGridSize     = b.claheTileGridSize;
            this.labBMin               = b.labBMin;
            this.labBMax               = b.labBMax;
            this.labLMinForYellow      = b.labLMinForYellow;
            this.hlsLMin               = b.hlsLMin;
            this.hlsSMax               = b.hlsSMax;
            this.grayWhiteMin          = b.grayWhiteMin;
            this.topHatKernelSize      = b.topHatKernelSize;
            this.topHatThreshold       = b.topHatThreshold;
            this.fusionMode            = b.fusionMode;
            this.openingKernelSize     = b.openingKernelSize;
            this.closingEnabled        = b.closingEnabled;
            this.closingKernelSize     = b.closingKernelSize;
            this.minComponentAreaRatio = b.minComponentAreaRatio;
            this.skeletonize           = b.skeletonize;
            this.skeletonMaxIterations = b.skeletonMaxIterations;
            this.temporalWindowSize    = b.temporalWindowSize;
            this.temporalMinAgreement  = b.temporalMinAgreement;
        }

        public static Config  defaults() { return new Builder().build(); }
        public static Builder builder()  { return new Builder(); }

        /** Mutable builder for {@link Config}. */
        public static final class Builder {
            // Preprocessing
            private int    gaussianKernelSize     = 5;
            private double claheClipLimit         = 2.0;
            private Size   claheTileGridSize      = new Size(8, 8);

            // Yellow / orange — loosened so faded yellow and orange construction-zone paint pass.
            private int    labBMin                = 135;
            private int    labBMax                = 255;
            private int    labLMinForYellow       = 70;

            // White / light-gray — loosened: dim or weathered paint and gray dashes survive.
            private int    hlsLMin                = 180;
            private int    hlsSMax                = 60;
            private int    grayWhiteMin           = 195;

            // Top-Hat — kernel sized to cover a typical paint width on dashcam footage;
            // threshold lowered so weaker local peaks (faded paint) still register.
            private Size   topHatKernelSize       = new Size(15, 15);
            private int    topHatThreshold        = 25;

            // Fusion — STRICT (AND) gives precise pixel-accurate output.
            private FusionMode fusionMode         = FusionMode.STRICT;

            // Cleanup — opening only; closing left off so we never artificially thicken lines.
            private Size    openingKernelSize     = new Size(3, 3);
            private boolean closingEnabled        = false;
            private Size    closingKernelSize     = new Size(3, 5);

            private double  minComponentAreaRatio = 0.00003;          // ~0.003 % of frame area

            // Skeletonization — off by default; turn on for pixel-thin centerlines.
            private boolean skeletonize           = false;
            private int     skeletonMaxIterations = 100;

            // Temporal — agreement of 2 out of 3 frames kills single-frame flicker.
            private int     temporalWindowSize    = 3;
            private int     temporalMinAgreement  = 2;

            public Builder gaussianKernelSize(int v)         { this.gaussianKernelSize = v; return this; }
            public Builder claheClipLimit(double v)          { this.claheClipLimit = v; return this; }
            public Builder claheTileGridSize(Size v)         { this.claheTileGridSize = v; return this; }
            public Builder labBMin(int v)                    { this.labBMin = v; return this; }
            public Builder labBMax(int v)                    { this.labBMax = v; return this; }
            public Builder labLMinForYellow(int v)           { this.labLMinForYellow = v; return this; }
            public Builder hlsLMin(int v)                    { this.hlsLMin = v; return this; }
            public Builder hlsSMax(int v)                    { this.hlsSMax = v; return this; }
            public Builder grayWhiteMin(int v)               { this.grayWhiteMin = v; return this; }
            public Builder topHatKernelSize(Size v)          { this.topHatKernelSize = v; return this; }
            public Builder topHatThreshold(int v)            { this.topHatThreshold = v; return this; }
            public Builder fusionMode(FusionMode v)          { this.fusionMode = v; return this; }
            public Builder openingKernelSize(Size v)         { this.openingKernelSize = v; return this; }
            public Builder closingEnabled(boolean v)         { this.closingEnabled = v; return this; }
            public Builder closingKernelSize(Size v)         { this.closingKernelSize = v; return this; }
            public Builder minComponentAreaRatio(double v)   { this.minComponentAreaRatio = v; return this; }
            public Builder skeletonize(boolean v)            { this.skeletonize = v; return this; }
            public Builder skeletonMaxIterations(int v)      { this.skeletonMaxIterations = v; return this; }
            public Builder temporalWindowSize(int v)         { this.temporalWindowSize = v; return this; }
            public Builder temporalMinAgreement(int v)       { this.temporalMinAgreement = v; return this; }

            public Config build() {
                if (gaussianKernelSize < 1 || gaussianKernelSize % 2 == 0)
                    throw new IllegalArgumentException("gaussianKernelSize must be a positive odd integer");
                if (claheClipLimit <= 0)
                    throw new IllegalArgumentException("claheClipLimit must be > 0");
                if (temporalWindowSize < 1)
                    throw new IllegalArgumentException("temporalWindowSize must be >= 1");
                if (temporalMinAgreement < 1 || temporalMinAgreement > temporalWindowSize)
                    throw new IllegalArgumentException("temporalMinAgreement must be in [1, temporalWindowSize]");
                if (minComponentAreaRatio < 0)
                    throw new IllegalArgumentException("minComponentAreaRatio must be >= 0");
                if (skeletonMaxIterations < 1)
                    throw new IllegalArgumentException("skeletonMaxIterations must be >= 1");
                Objects.requireNonNull(fusionMode, "fusionMode");
                return new Config(this);
            }
        }
    }

    // =========================================================================================
    // State
    // =========================================================================================

    private final Config config;
    private final CLAHE  clahe;

    private final Mat openingKernel;
    private final Mat closingKernel;
    private final Mat topHatKernel;
    private final Mat skeletonKernel;          // 3x3 cross, used iteratively by skeletonize()

    private final Deque<Mat> recentMasks = new ArrayDeque<>();

    private int lastWidth  = -1;
    private int lastHeight = -1;
    private boolean released = false;

    // =========================================================================================
    // Construction
    // =========================================================================================

    public LaneMarkingMaskExtractor() {
        this(Config.defaults());
    }

    public LaneMarkingMaskExtractor(Config config) {
        this.config         = Objects.requireNonNull(config, "config");
        this.clahe          = Imgproc.createCLAHE(config.claheClipLimit, config.claheTileGridSize);
        this.openingKernel  = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, config.openingKernelSize);
        this.closingKernel  = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,    config.closingKernelSize);
        this.topHatKernel   = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,    config.topHatKernelSize);
        this.skeletonKernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,   new Size(3, 3));
    }

    // =========================================================================================
    // Public API
    // =========================================================================================

    /**
     * Process a single BGR frame and return a binary mask of detected lane-marking pixels.
     *
     * @param frame three-channel CV_8UC3 BGR image; not modified
     * @return new CV_8UC1 mask where 255 marks lane-marking pixels and 0 marks background;
     *         the caller owns the returned {@code Mat} and must release it.
     */
    public Mat process(Mat frame) {
        if (released) throw new IllegalStateException("Extractor has been released");
        Objects.requireNonNull(frame, "frame");
        if (frame.empty())       throw new IllegalArgumentException("Input frame is empty");
        if (frame.channels() != 3)
            throw new IllegalArgumentException(
                    "Expected 3-channel BGR image, got " + frame.channels() + " channels");

        if (frame.width() != lastWidth || frame.height() != lastHeight) {
            clearRecentMasks();
            lastWidth  = frame.width();
            lastHeight = frame.height();
        }

        Mat blurred       = new Mat();
        Mat lab           = new Mat();
        Mat enhanced      = new Mat();
        Mat hls           = new Mat();
        Mat gray          = new Mat();
        Mat bMask         = new Mat();
        Mat lGate         = new Mat();
        Mat yellowMask    = new Mat();
        Mat whiteMaskHls  = new Mat();
        Mat whiteMaskGray = new Mat();
        Mat whiteMask     = new Mat();
        Mat colorMask     = new Mat();
        Mat topHat        = new Mat();
        Mat topHatMask    = new Mat();
        Mat fused         = new Mat();
        Mat opened        = new Mat();
        Mat cleaned       = new Mat();
        Mat areaFiltered  = null;
        Mat afterSkeleton = null;
        List<Mat> labChannels = new ArrayList<>(3);

        try {
            // -- 1. Denoise --------------------------------------------------------------------
            int k = config.gaussianKernelSize;
            Imgproc.GaussianBlur(frame, blurred, new Size(k, k), 0);

            // -- 2. CLAHE on L of LAB ----------------------------------------------------------
            Imgproc.cvtColor(blurred, lab, Imgproc.COLOR_BGR2Lab);
            Core.split(lab, labChannels);
            Mat lChannel = labChannels.get(0);
            Mat bChannel = labChannels.get(2);
            clahe.apply(lChannel, lChannel);
            Core.merge(labChannels, lab);
            Imgproc.cvtColor(lab, enhanced, Imgproc.COLOR_Lab2BGR);

            // -- 3a. Yellow: high B + L not too dark -------------------------------------------
            Core.inRange(bChannel, new Scalar(config.labBMin), new Scalar(config.labBMax), bMask);
            Imgproc.threshold(lChannel, lGate, config.labLMinForYellow, 255, Imgproc.THRESH_BINARY);
            Core.bitwise_and(bMask, lGate, yellowMask);

            // -- 3b. White: HLS (high L, low S) OR very bright grayscale -----------------------
            Imgproc.cvtColor(enhanced, hls, Imgproc.COLOR_BGR2HLS);
            Core.inRange(hls,
                    new Scalar(0,   config.hlsLMin, 0),
                    new Scalar(180, 255,            config.hlsSMax),
                    whiteMaskHls);
            Imgproc.cvtColor(enhanced, gray, Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(gray, whiteMaskGray, config.grayWhiteMin, 255, Imgproc.THRESH_BINARY);
            Core.bitwise_or(whiteMaskHls, whiteMaskGray, whiteMask);

            // -- 3c. Colour evidence = yellow ∪ white ------------------------------------------
            Core.bitwise_or(yellowMask, whiteMask, colorMask);

            // -- 4. Structural evidence: Top-Hat → threshold -----------------------------------
            // Top-Hat = original − opening(original). Highlights bright structures smaller than
            // the kernel — lane paint by design. Large bright surfaces (cars, concrete, sky)
            // are flattened by the opening and disappear from the result.
            Imgproc.morphologyEx(gray, topHat, Imgproc.MORPH_TOPHAT, topHatKernel);
            Imgproc.threshold(topHat, topHatMask, config.topHatThreshold, 255, Imgproc.THRESH_BINARY);

            // -- 5. Fusion ---------------------------------------------------------------------
            if (config.fusionMode == FusionMode.STRICT) {
                // Pixel must be lane-coloured AND a local bright peak. High precision.
                Core.bitwise_and(colorMask, topHatMask, fused);
            } else {
                // Either signal is enough. Higher recall on faded paint, more false positives.
                Core.bitwise_or(colorMask, topHatMask, fused);
            }

            // -- 6. Light cleanup --------------------------------------------------------------
            Imgproc.morphologyEx(fused, opened, Imgproc.MORPH_OPEN, openingKernel);
            if (config.closingEnabled) {
                Imgproc.morphologyEx(opened, cleaned, Imgproc.MORPH_CLOSE, closingKernel);
            } else {
                opened.copyTo(cleaned);
            }

            // -- 7. Drop tiny components -------------------------------------------------------
            areaFiltered = (config.minComponentAreaRatio > 0)
                    ? removeSmallComponents(cleaned, frame.width() * frame.height())
                    : cleaned.clone();

            // -- 8. Optional skeletonization for pixel-thin centerlines ------------------------
            Mat forTemporal;
            if (config.skeletonize) {
                afterSkeleton = skeletonize(areaFiltered, skeletonKernel, config.skeletonMaxIterations);
                forTemporal   = afterSkeleton;
            } else {
                forTemporal   = areaFiltered;
            }

            // -- 9. Temporal agreement ---------------------------------------------------------
            return applyTemporalSmoothing(forTemporal);

        } finally {
            blurred.release();
            lab.release();
            enhanced.release();
            hls.release();
            gray.release();
            bMask.release();
            lGate.release();
            yellowMask.release();
            whiteMaskHls.release();
            whiteMaskGray.release();
            whiteMask.release();
            colorMask.release();
            topHat.release();
            topHatMask.release();
            fused.release();
            opened.release();
            cleaned.release();
            if (areaFiltered  != null) areaFiltered.release();
            if (afterSkeleton != null) afterSkeleton.release();
            for (Mat ch : labChannels) ch.release();
        }
    }

    /** Flush the temporal buffer. Call between unrelated videos / after a scene cut. */
    public void reset() {
        clearRecentMasks();
        lastWidth  = -1;
        lastHeight = -1;
    }

    /** Free all native resources. Safe to call multiple times. */
    public void release() {
        if (released) return;
        clearRecentMasks();
        openingKernel.release();
        closingKernel.release();
        topHatKernel.release();
        skeletonKernel.release();
        released = true;
    }

    // =========================================================================================
    // Helpers
    // =========================================================================================

    /** Drop connected components whose area is below {@code minComponentAreaRatio × totalArea}. */
    private Mat removeSmallComponents(Mat binary, int totalArea) {
        double minArea = totalArea * config.minComponentAreaRatio;

        Mat labels    = new Mat();
        Mat stats     = new Mat();
        Mat centroids = new Mat();
        Mat result    = Mat.zeros(binary.size(), binary.type());
        Mat tmp       = new Mat();
        try {
            int nLabels = Imgproc.connectedComponentsWithStats(
                    binary, labels, stats, centroids, 8, CvType.CV_32S);
            for (int label = 1; label < nLabels; label++) {       // label 0 = background
                double area = stats.get(label, Imgproc.CC_STAT_AREA)[0];
                if (area >= minArea) {
                    Core.compare(labels, new Scalar(label), tmp, Core.CMP_EQ);
                    Core.bitwise_or(result, tmp, result);
                }
            }
            return result;
        } finally {
            labels.release();
            stats.release();
            centroids.release();
            tmp.release();
        }
    }

    /**
     * Classic morphological skeletonization: repeatedly accumulate the residual
     * <code>img − opening(img)</code>, then erode, until the image vanishes. Output is a binary
     * mask of single-pixel-wide medial axes.
     */
    private static Mat skeletonize(Mat binary, Mat element, int maxIterations) {
        Mat skeleton = Mat.zeros(binary.size(), CvType.CV_8UC1);
        Mat working  = binary.clone();
        Mat eroded   = new Mat();
        Mat opened   = new Mat();
        Mat residual = new Mat();
        try {
            for (int i = 0; i < maxIterations; i++) {
                Imgproc.erode(working, eroded, element);
                Imgproc.dilate(eroded, opened, element);          // opening = erode ∘ dilate
                Core.subtract(working, opened, residual);
                Core.bitwise_or(skeleton, residual, skeleton);
                eroded.copyTo(working);
                if (Core.countNonZero(working) == 0) break;
            }
            return skeleton;
        } finally {
            working.release();
            eroded.release();
            opened.release();
            residual.release();
        }
    }

    /**
     * Keep a pixel only if it appears in at least {@code temporalMinAgreement} of the last
     * {@code temporalWindowSize} frames. During warm-up the current mask is passed through.
     */
    private Mat applyTemporalSmoothing(Mat currentMask) {
        if (config.temporalWindowSize <= 1) {
            return currentMask.clone();
        }

        recentMasks.addLast(currentMask.clone());
        while (recentMasks.size() > config.temporalWindowSize) {
            Mat removed = recentMasks.pollFirst();
            if (removed != null) removed.release();
        }
        if (recentMasks.size() < config.temporalMinAgreement) {
            return currentMask.clone();
        }

        Mat accumulator = Mat.zeros(currentMask.size(), CvType.CV_16UC1);
        Mat converted   = new Mat();
        try {
            for (Mat m : recentMasks) {
                m.convertTo(converted, CvType.CV_16UC1, 1.0 / 255.0);
                Core.add(accumulator, converted, accumulator);
            }
            Mat result = new Mat();
            Core.compare(accumulator,
                    new Scalar(config.temporalMinAgreement - 1),
                    result, Core.CMP_GT);
            return result;                                       // CV_8U with {0, 255}
        } finally {
            accumulator.release();
            converted.release();
        }
    }

    private void clearRecentMasks() {
        while (!recentMasks.isEmpty()) {
            Mat m = recentMasks.pollFirst();
            if (m != null) m.release();
        }
    }
}
