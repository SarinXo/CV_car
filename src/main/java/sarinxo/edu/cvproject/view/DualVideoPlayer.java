package sarinxo.edu.cvproject.view;

import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
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
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import sarinxo.edu.cvproject.detection.LaneCurveDetector;
import sarinxo.edu.cvproject.detection.LaneMarkingMaskExtractor;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class DualVideoPlayer {

    private MediaPlayer player1;

    private MediaView view1;
    private ImageView view2; // обработанное видео
    private ImageView view3; // бинарная маска

    private Slider seek1 = new Slider();
    private Slider volume1 = new Slider(0, 1, 0.5);
    private Label time1 = new Label("00:00 / 00:00");

    private Button playButton;
    private Button pauseButton;
    private Button stopButton;

    private List<Image> processedVideoFrames = new ArrayList<>();
    private List<Image> maskVideoFrames = new ArrayList<>();

    public void start(Stage stage) {

        view1 = new MediaView();
        view2 = new ImageView();
        view3 = new ImageView();

        view1.setPreserveRatio(true);
        view2.setPreserveRatio(true);
        view3.setPreserveRatio(true);

        StackPane videoPane1 = new StackPane(view1);
        StackPane videoPane2 = new StackPane(view2);
        StackPane videoPane3 = new StackPane(view3);

        String paneStyle =
                "-fx-background-color: #0d0d0f;" +
                "-fx-background-radius: 8;" +
                "-fx-border-color: #2a2a2e;" +
                "-fx-border-width: 1;" +
                "-fx-border-radius: 8;";
        for (StackPane p : new StackPane[]{videoPane1, videoPane2, videoPane3}) {
            p.setStyle(paneStyle);
            p.setMinSize(0, 0);
            // Ширина плитки = высота × 16/9. Видео всегда занимает максимум
            // доступной высоты и пропорционально растёт по ширине.
            p.prefWidthProperty().bind(p.heightProperty().multiply(16.0 / 9.0));
        }

        enableDragAndDrop(videoPane1);

        VBox controls1 = createControls();

        VBox left  = buildCard("Оригинал",  videoPane1, controls1);
        VBox right = buildCard("Результат", videoPane2, null);
        VBox third = buildCard("Маска",     videoPane3, null);

        VBox.setVgrow(videoPane1, Priority.ALWAYS);
        VBox.setVgrow(videoPane2, Priority.ALWAYS);
        VBox.setVgrow(videoPane3, Priority.ALWAYS);

        HBox videos = new HBox(12, left, right, third);
        for (VBox card : new VBox[]{left, right, third}) {
            card.setMinWidth(VBox.USE_PREF_SIZE);
        }

        ScrollPane scroll = new ScrollPane(videos);
        scroll.setFitToHeight(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.AS_NEEDED);
        scroll.setVbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.setStyle(
                "-fx-background: #18181b;" +
                "-fx-background-color: #18181b;" +
                "-fx-padding: 0;");

        VBox root = new VBox(scroll);
        root.setPadding(new Insets(12));
        root.setStyle("-fx-background-color: #18181b;");
        VBox.setVgrow(scroll, Priority.ALWAYS);

        Scene scene = new Scene(root, 1100, 620);
        stage.setTitle("Lane Detection Player");
        stage.setScene(scene);
        stage.show();

        double minWidth = 160, minHeight = 60;
        bindAutoScale(view1, videoPane1, minWidth, minHeight);
        bindAutoScale(view2, videoPane2, minWidth, minHeight);
        bindAutoScale(view3, videoPane3, minWidth, minHeight);
    }

    private VBox buildCard(String title, StackPane preview, VBox controls) {
        Label header = new Label(title);
        header.setStyle(
                "-fx-text-fill: #d4d4d8;" +
                "-fx-font-size: 12px;" +
                "-fx-font-weight: bold;" +
                "-fx-padding: 0 0 0 4;");

        VBox card = (controls == null)
                ? new VBox(6, header, preview)
                : new VBox(6, header, preview, controls);
        card.setStyle(
                "-fx-background-color: #1f1f23;" +
                "-fx-background-radius: 10;" +
                "-fx-padding: 10;");
        return card;
    }

    private void bindAutoScale(MediaView view, StackPane pane, double minW, double minH) {
        view.fitWidthProperty().bind(Bindings.createDoubleBinding(
                () -> Math.max(minW, pane.getWidth()),
                pane.widthProperty()));
        view.fitHeightProperty().bind(Bindings.createDoubleBinding(
                () -> Math.max(minH, pane.getHeight()),
                pane.heightProperty()));
    }
    private void bindAutoScale(ImageView view, StackPane pane, double minW, double minH) {
        view.fitWidthProperty().bind(Bindings.createDoubleBinding(
                () -> Math.max(minW, pane.getWidth()),
                pane.widthProperty()));
        view.fitHeightProperty().bind(Bindings.createDoubleBinding(
                () -> Math.max(minH, pane.getHeight()),
                pane.heightProperty()));
    }

    private VBox createControls() {
        playButton  = styledButton("▶");
        pauseButton = styledButton("⏸");
        stopButton  = styledButton("⏹");

        Label volumeIcon = new Label("🔊");
        volumeIcon.setStyle("-fx-text-fill: #d4d4d8;");

        HBox buttons = new HBox(8, playButton, pauseButton, stopButton, volumeIcon, volume1);
        buttons.setAlignment(Pos.CENTER_LEFT);

        time1.setStyle("-fx-text-fill: #a1a1aa; -fx-font-size: 11px;");

        VBox box = new VBox(6, seek1, buttons, time1);
        box.setPadding(new Insets(6, 2, 2, 2));

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

    private Button styledButton(String text) {
        Button b = new Button(text);
        b.setStyle(
                "-fx-background-color: #27272a;" +
                "-fx-text-fill: #f4f4f5;" +
                "-fx-background-radius: 6;" +
                "-fx-padding: 4 12 4 12;" +
                "-fx-font-size: 13px;");
        b.setOnMouseEntered(e -> b.setStyle(
                "-fx-background-color: #3f3f46;" +
                "-fx-text-fill: #ffffff;" +
                "-fx-background-radius: 6;" +
                "-fx-padding: 4 12 4 12;" +
                "-fx-font-size: 13px;"));
        b.setOnMouseExited(e -> b.setStyle(
                "-fx-background-color: #27272a;" +
                "-fx-text-fill: #f4f4f5;" +
                "-fx-background-radius: 6;" +
                "-fx-padding: 4 12 4 12;" +
                "-fx-font-size: 13px;"));
        return b;
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
        List<Image> maskFrames = new ArrayList<>();
        LaneMarkingMaskExtractor maskExtractor = new LaneMarkingMaskExtractor();
        LaneCurveDetector detector = new LaneCurveDetector();
        while (capture.read(frame)) {
            if (frame.empty()) continue;
            Mat mask = maskExtractor.process(frame);//тута
            Image maskImage = matToImage(mask);
            Mat processed = detector.process(frame, mask);//здеся
            Image fxImage = matToImage(processed);
            processedFrames.add(fxImage);
            maskFrames.add(maskImage);
            mask.release();
            processed.release();
        }

        capture.release();
        frame.release();
        maskExtractor.release();

        if (processedFrames.isEmpty()) return;

        Platform.runLater(() -> {
            processedVideoFrames = processedFrames;
            maskVideoFrames = maskFrames;
            setControlsDisabled(false);

            AnimationTimer timer = new AnimationTimer() {
                @Override
                public void handle(long now) {
                    if (player1 == null || player1.getStatus() != MediaPlayer.Status.PLAYING) return;

                    double currentTime = player1.getCurrentTime().toSeconds();
                    int frameIndex = (int) (currentTime * 30);
                    if (frameIndex >= processedVideoFrames.size()) frameIndex = processedVideoFrames.size() - 1;

                    view2.setImage(processedVideoFrames.get(frameIndex));
                    if (frameIndex < maskVideoFrames.size()) {
                        view3.setImage(maskVideoFrames.get(frameIndex));
                    }
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

    private Image matToImage(Mat mat) {
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", mat, buffer);
        return new Image(new ByteArrayInputStream(buffer.toArray()));
    }
}