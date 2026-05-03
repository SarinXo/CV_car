package sarinxo.edu.cvproject.view;

import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.media.Media;
import javafx.scene.media.MediaPlayer;
import javafx.scene.media.MediaView;
import javafx.stage.Stage;
import javafx.util.Duration;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import sarinxo.edu.cvproject.LaneDetector;
import sarinxo.edu.cvproject.LaneDetector2;
import sarinxo.edu.cvproject.detection.LaneDetector3;
import sarinxo.edu.cvproject.detection.PerspectiveInitializer;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class DualVideoPlayer {

    private MediaPlayer player1;

    private MediaView view1;
    private ImageView view2; // обработанное видео

    private Slider seek1 = new Slider();
    private Slider volume1 = new Slider(0, 1, 0.5);
    private Label time1 = new Label("00:00 / 00:00");

    private Button playButton;
    private Button pauseButton;
    private Button stopButton;

    private List<Image> processedVideoFrames = new ArrayList<>();

    public void start(Stage stage) {

        view1 = new MediaView();
        view2 = new ImageView();

        view1.setPreserveRatio(true);
        view2.setPreserveRatio(true);

        StackPane videoPane1 = new StackPane(view1);
        StackPane videoPane2 = new StackPane(view2);

        videoPane1.setStyle("-fx-border-color: #444; -fx-border-width: 2;");
        videoPane2.setStyle("-fx-border-color: #444; -fx-border-width: 2;");

        enableDragAndDrop(videoPane1);

        VBox controls1 = createControls();

        VBox left = new VBox(8, videoPane1, controls1);
        VBox right = new VBox(8, videoPane2);

        VBox.setVgrow(videoPane1, Priority.ALWAYS);
        VBox.setVgrow(videoPane2, Priority.ALWAYS);

        HBox videos = new HBox(15, left, right);
        HBox.setHgrow(left, Priority.ALWAYS);
        HBox.setHgrow(right, Priority.ALWAYS);

        VBox root = new VBox(15, videos);
        root.setPadding(new Insets(15));
        VBox.setVgrow(videos, Priority.ALWAYS);

        Scene scene = new Scene(root, 900, 600);
        stage.setTitle("Dual Video Player with Lane Detection");
        stage.setScene(scene);
        stage.show();

        double minWidth = 200, minHeight = 112, maxWidth = 600, maxHeight = 337;
        bindAutoScale(view1, videoPane1, controls1, minWidth, minHeight, maxWidth, maxHeight);
        bindAutoScale(view2, videoPane2, null, minWidth, minHeight, maxWidth, maxHeight);
    }

    private void bindAutoScale(MediaView view, StackPane pane, VBox controls, double minW, double minH, double maxW, double maxH) {
        view.fitWidthProperty().bind(Bindings.createDoubleBinding(
                () -> Math.max(minW, Math.min(pane.getWidth(), maxW)),
                pane.widthProperty()));
        if (controls != null) {
            view.fitHeightProperty().bind(Bindings.createDoubleBinding(
                    () -> Math.max(minH, Math.min(pane.getHeight() - controls.getHeight(), maxH)),
                    pane.heightProperty(), controls.heightProperty()));
        } else {
            view.fitHeightProperty().bind(Bindings.createDoubleBinding(
                    () -> Math.max(minH, Math.min(pane.getHeight(), maxH)),
                    pane.heightProperty()));
        }
    }
    private void bindAutoScale(ImageView view, StackPane pane, VBox controls, double minW, double minH, double maxW, double maxH) {
        view.fitWidthProperty().bind(Bindings.createDoubleBinding(
                () -> Math.max(minW, Math.min(pane.getWidth(), maxW)),
                pane.widthProperty()));
        if (controls != null) {
            view.fitHeightProperty().bind(Bindings.createDoubleBinding(
                    () -> Math.max(minH, Math.min(pane.getHeight() - controls.getHeight(), maxH)),
                    pane.heightProperty(), controls.heightProperty()));
        } else {
            view.fitHeightProperty().bind(Bindings.createDoubleBinding(
                    () -> Math.max(minH, Math.min(pane.getHeight(), maxH)),
                    pane.heightProperty()));
        }
    }

    private VBox createControls() {
        playButton = new Button("▶");
        pauseButton = new Button("⏸");
        stopButton = new Button("⏹");

        HBox buttons = new HBox(5, playButton, pauseButton, stopButton, new Label("🔊"), volume1);
        buttons.setAlignment(Pos.CENTER_LEFT);

        VBox box = new VBox(5, seek1, buttons, time1);
        box.setPadding(new Insets(5));

        playButton.setOnAction(e -> player1.play());
        pauseButton.setOnAction(e -> player1.pause());
        stopButton.setOnAction(e -> player1.stop());

        volume1.valueProperty().addListener((obs, o, n) -> {
            if (player1 != null) player1.setVolume(n.doubleValue());
        });

        seek1.valueProperty().addListener((obs, o, n) -> {
            if (player1 != null && seek1.isValueChanging()) {
                player1.seek(Duration.seconds(n.doubleValue()));
            }
        });

        return box;
    }

    private void bindPlayer(MediaPlayer player) {
        player.setOnReady(() -> {
            Duration total = player.getTotalDuration();
            seek1.setMax(total.toSeconds());
        });

        player.currentTimeProperty().addListener((obs, oldTime, newTime) -> {
            if (!seek1.isValueChanging()) seek1.setValue(newTime.toSeconds());
            updateTimeLabel(time1, newTime, player.getTotalDuration());
        });
    }

    private void updateTimeLabel(Label label, Duration current, Duration total) {
        label.setText(format(current) + " / " + format(total));
    }

    private String format(Duration d) {
        int sec = (int) d.toSeconds();
        int m = sec / 60;
        int s = sec % 60;
        return String.format("%02d:%02d", m, s);
    }

    private void enableDragAndDrop(StackPane pane) {
        pane.setOnDragOver(e -> {
            if (e.getDragboard().hasFiles()) e.acceptTransferModes(TransferMode.COPY);
            e.consume();
        });

        pane.setOnDragDropped(e -> {
            Dragboard db = e.getDragboard();
            if (db.hasFiles()) {
                File file = db.getFiles().get(0);
                loadVideo(file);
            }
            e.setDropCompleted(true);
            e.consume();
        });
    }

    private void loadVideo(File file) {
        // Блокируем кнопки управления
        setControlsDisabled(true);

        if (player1 != null) player1.dispose();
        Media media = new Media(file.toURI().toString());
        player1 = new MediaPlayer(media);
        view1.setMediaPlayer(player1);
        bindPlayer(player1);

        new Thread(() -> startProcessing(file)).start();
    }

    private void startProcessing(File file) {
        VideoCapture capture = new VideoCapture(file.getAbsolutePath());
        if (!capture.isOpened()) return;

        Mat frame = new Mat();
        List<Image> processedFrames = new ArrayList<>();
        LaneDetector3 detector = null;
        while (capture.read(frame)) {
            if (frame.empty()) continue;
            if (detector == null) {
                PerspectiveInitializer.Result initialize = new PerspectiveInitializer().initialize(frame);
                detector = new LaneDetector3(initialize.src, initialize.dst);
            }
            Mat processed = detector.processFrame(frame);//тута
            Image fxImage = matToImage(processed);
            processedFrames.add(fxImage);
        }

        capture.release();
        frame.release();

        if (processedFrames.isEmpty()) return;

        Platform.runLater(() -> {
            processedVideoFrames = processedFrames;
            setControlsDisabled(false);

            AnimationTimer timer = new AnimationTimer() {
                @Override
                public void handle(long now) {
                    if (player1 == null || player1.getStatus() != MediaPlayer.Status.PLAYING) return;

                    double currentTime = player1.getCurrentTime().toSeconds();
                    int frameIndex = (int) (currentTime * 30);
                    if (frameIndex >= processedVideoFrames.size()) frameIndex = processedVideoFrames.size() - 1;

                    view2.setImage(processedVideoFrames.get(frameIndex));
                }
            };
            timer.start();
        });
    }

    private void setControlsDisabled(boolean disabled) {
        playButton.setDisable(disabled);
        pauseButton.setDisable(disabled);
        stopButton.setDisable(disabled);
        seek1.setDisable(disabled);
        volume1.setDisable(disabled);
    }

    //todo delete
    private Mat detectLanes(Mat frame) {
        Mat hsv = new Mat();
        Mat maskWhite = new Mat();
        Mat maskYellow = new Mat();
        Mat mask = new Mat();

        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);
        Core.inRange(hsv, new Scalar(0, 0, 200), new Scalar(180, 30, 255), maskWhite);
        Core.inRange(hsv, new Scalar(15, 100, 100), new Scalar(35, 255, 255), maskYellow);
        Core.addWeighted(maskWhite, 1.0, maskYellow, 1.0, 0, mask);

        Mat edges = new Mat();
        Imgproc.GaussianBlur(mask, mask, new Size(5, 5), 0);
        Imgproc.Canny(mask, edges, 50, 150);

        int w = edges.width();
        int h = edges.height();
        Mat maskedEdges = Mat.zeros(edges.size(), edges.type());
        Point p1 = new Point(0, h), p2 = new Point(w, h), p3 = new Point(w * 0.9, h * 0.6), p4 = new Point(w * 0.1, h * 0.6);
        MatOfPoint poly = new MatOfPoint(p1, p2, p3, p4);
        List<MatOfPoint> polys = new ArrayList<>();
        polys.add(poly);
        Imgproc.fillPoly(maskedEdges, polys, new Scalar(255));
        Core.bitwise_and(edges, maskedEdges, maskedEdges);

        Mat lines = new Mat();
        Imgproc.HoughLinesP(maskedEdges, lines, 1, Math.PI / 180, 50, 20, 10);

        Mat output = frame.clone();
        double centerX = w / 2.0;

        List<double[]> leftLines = new ArrayList<>();
        List<double[]> rightLines = new ArrayList<>();

        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);
            double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
            double slope = (y2 - y1) / (x2 - x1 + 1e-6);
            if (Math.abs(slope) < 0.4) continue;

            if (slope < 0 && x1 < centerX && x2 < centerX) leftLines.add(l);
            else if (slope > 0 && x1 > centerX && x2 > centerX) rightLines.add(l);
        }

        drawSegments(output, leftLines, new Scalar(255, 100, 0));
        drawSegments(output, rightLines, new Scalar(0, 100, 255));

        hsv.release();
        maskWhite.release();
        maskYellow.release();
        mask.release();
        edges.release();
        maskedEdges.release();
        lines.release();
        poly.release();

        return output;
    }

    private void drawSegments(Mat img, List<double[]> lines, Scalar color) {
        for (double[] l : lines) {
            Imgproc.line(img, new Point(l[0], l[1]), new Point(l[2], l[3]), color, 4);
        }
    }

    private Image matToImage(Mat mat) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", mat, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }
}