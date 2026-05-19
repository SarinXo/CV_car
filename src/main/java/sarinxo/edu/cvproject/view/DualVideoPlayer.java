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
import javafx.stage.Stage;
import javafx.util.Duration;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import sarinxo.edu.cvproject.detection.LaneCurveDetector;
import sarinxo.edu.cvproject.detection.LaneMarkingMaskExtractor;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;


public class DualVideoPlayer {

    private MediaPlayer player1;          // только как clock + audio; видео НЕ рендерится через MediaView

    private ImageView view1; // оригинальное видео (кадры от OpenCV/FFmpeg, а не от JavaFX-декодера)
    private ImageView view2; // обработанное видео
    private ImageView view3; // бинарная маска

    private Slider seek1 = new Slider();
    private Slider volume1 = new Slider(0, 1, 0.5);
    private Label time1 = new Label("00:00 / 00:00");

    private Button playButton;
    private Button pauseButton;
    private Button stopButton;

    private List<Image> originalVideoFrames  = new ArrayList<>();
    private List<Image> processedVideoFrames = new ArrayList<>();
    private List<Image> maskVideoFrames      = new ArrayList<>();

    // FPS захватывается из VideoCapture при обработке и используется в AnimationTimer
    // для расчёта frameIndex. 30.0 — безопасный дефолт на случай, если FPS неизвестен.
    private double currentFps = 30.0;

    // Активный AnimationTimer для view2/view3. Хранится, чтобы остановить предыдущий
    // при загрузке нового видео — иначе старый таймер продолжает обращаться к
    // замещённому player1 и устаревшим спискам кадров.
    private AnimationTimer activeTimer;

    // Пока true — пользователь взаимодействует со слайдером (нажал/тянет/только что
    // отпустил). currentTimeProperty-listener в этот момент НЕ перезаписывает значение
    // слайдера, иначе позиция, на которой пользователь кликнул, будет затёрта временем
    // плеера ещё до того, как onMouseReleased успеет прочитать её и вызвать seek().
    private boolean sliderUserInteracting = false;

    public void start(Stage stage) {

        view1 = new ImageView();
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

        // Play: если плеер дошёл до конца — перематываем на начало перед запуском.
        // Это позволяет пересматривать видео нажатием Play, без отдельного действия.
        playButton.setOnAction(e -> {
            if (player1 == null) return;
            Duration total   = player1.getTotalDuration();
            Duration current = player1.getCurrentTime();
            if (total != null && current != null
                    && !total.isUnknown()
                    && current.greaterThanOrEqualTo(total)) {
                player1.seek(Duration.ZERO);
            }
            player1.play();
        });
        pauseButton.setOnAction(e -> { if (player1 != null) player1.pause(); });
        // Stop: MediaPlayer.stop() сам сбрасывает currentTime в 0, и AnimationTimer
        // (после удаления PLAYING-гарда ниже) синхронно покажет нулевой кадр в Result/Mask.
        stopButton.setOnAction(e -> { if (player1 != null) player1.stop(); });

        volume1.valueProperty().addListener((obs, o, n) -> {
            if (player1 != null) player1.setVolume(n.doubleValue());
        });

        // Перетаскивание ползунка: пока пользователь тянет, JavaFX сам выставляет
        // isValueChanging=true, и listener мгновенно перематывает плеер за курсором —
        // получается «live preview». На release isValueChanging уходит в false, поэтому
        // выполнение завершит onMouseReleased ниже.
        seek1.valueProperty().addListener((obs, o, n) -> {
            if (seek1.isValueChanging()) seekValidated(n.doubleValue());
        });
        // Клик по треку слайдера (без drag): JavaFX часто не выставляет isValueChanging,
        // поэтому полагаемся на mouse-pressed/released. Между ними time-listener не
        // перезаписывает позицию слайдера (см. флаг sliderUserInteracting).
        seek1.setOnMousePressed(e -> sliderUserInteracting = true);
        seek1.setOnMouseReleased(e -> {
            double target = seek1.getValue();
            sliderUserInteracting = false;
            seekValidated(target);
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
            // Только валидные продолжительности доходят до setMax — без этого слайдер
            // мог остаться с дефолтным max=100, и любое значение перетаскивания за реальную
            // длительность приводило к seek в конец / undefined-поведению.
            if (total != null && !total.isUnknown() && total.toSeconds() > 0) {
                seek1.setMax(total.toSeconds());
            }
        });

        player.currentTimeProperty().addListener((obs, oldTime, newTime) -> {
            // НЕ перезаписываем значение слайдера, пока:
            //   - пользователь тянет ползунок (isValueChanging)
            //   - пользователь только что кликнул на трек (sliderUserInteracting между
            //     mouse-pressed и mouse-released)
            // Иначе time-listener затирает позицию пользователя ещё до того, как
            // onMouseReleased прочитает её и сделает seek.
            if (!seek1.isValueChanging() && !sliderUserInteracting) {
                seek1.setValue(newTime.toSeconds());
            }
            updateTimeLabel(time1, newTime, player.getTotalDuration());
        });
    }

    /**
     * Перематывает {@code player1} в {@code seconds}, клампя значение в реальный диапазон
     * [0, totalDuration]. Любой невалидный target (NaN, отрицательный, без известной
     * длительности) отбрасывается. Это страховка от трёх типов багов:
     * <ul>
     *   <li>слайдер ещё не получил setMax() — может выдать значение > длительности,</li>
     *   <li>time-listener успел перебить значение слайдера до setOnMouseReleased,</li>
     *   <li>пользователь как-то протащил ползунок за пределы [0, totalDuration].</li>
     * </ul>
     */
    private void seekValidated(double seconds) {
        if (player1 == null) return;
        if (Double.isNaN(seconds) || Double.isInfinite(seconds)) return;
        Duration total = player1.getTotalDuration();
        if (total == null || total.isUnknown() || total.toSeconds() <= 0) return;
        double clamped = Math.max(0.0, Math.min(total.toSeconds(), seconds));
        player1.seek(Duration.seconds(clamped));
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

        // Останавливаем предыдущий синхронизатор — иначе он продолжит обращаться
        // к замещённому player1 и устаревшим спискам кадров.
        if (activeTimer != null) {
            activeTimer.stop();
            activeTimer = null;
        }

        if (player1 != null) player1.dispose();
        Media media = new Media(file.toURI().toString());
        player1 = new MediaPlayer(media);
        // ВАЖНО: видео НЕ привязывается к MediaView. JavaFX-декодер плохо ладит с
        // некоторыми форматами — кадры пропускаются, отдельные участки не сеекаются.
        // Вместо этого все три view рендерятся из кадров, прочитанных OpenCV (тот же
        // декодер, что используется для обработки), синхронно по player1.getCurrentTime().
        // MediaPlayer оставлен ради аудиодорожки и как источник часов воспроизведения.
        bindPlayer(player1);

        // Когда видео доиграло, JavaFX оставляет плеер в STOPPED/PAUSED на последнем кадре.
        // Явно ставим pause(), чтобы статус был предсказуемым, а handler кнопки Play
        // мог детектировать «мы у конца» и перемотать на начало.
        player1.setOnEndOfMedia(() -> {
            if (player1 != null) player1.pause();
        });

        new Thread(() -> {
            try {
                startProcessing(file);
            } catch (Throwable t) {
                System.err.println("[startProcessing] Failed: " + t);
                t.printStackTrace();
                Platform.runLater(() -> setControlsDisabled(false));
            }
        }).start();
    }

    private void startProcessing(File file) {
        VideoCapture capture = new VideoCapture(file.getAbsolutePath());
        if (!capture.isOpened()) {
            System.err.println("[startProcessing] VideoCapture failed to open: " + file);
            return;
        }

        // Берём фактический FPS видео из контейнера. Для расчёта frameIndex в таймере
        // нужно реальное значение, иначе на видео с FPS ≠ 30 Result/Mask «уплывают»
        // вперёд или отстают от оригинала.
        double detectedFps = capture.get(org.opencv.videoio.Videoio.CAP_PROP_FPS);
        if (detectedFps <= 0 || Double.isNaN(detectedFps) || Double.isInfinite(detectedFps)) {
            detectedFps = 30.0;
        }
        final double fps = detectedFps;

        Mat frame = new Mat();
        List<Image> originalFrames  = new ArrayList<>();
        List<Image> processedFrames = new ArrayList<>();
        List<Image> maskFrames      = new ArrayList<>();
        // Pipeline: BGR кадр → v1 (color/edge extractor) → бинарная маска со шумом →
        // v2 (PCA-фильтр компонент) → бинарная маска только с прямолинейными
        // штрихами. v2 работает только с готовой маской, поэтому v1 обязателен.
        LaneMarkingMaskExtractor maskExtractor  = new LaneMarkingMaskExtractor();
        LaneCurveDetector detector = new LaneCurveDetector();
        int frameIdx = 0;
        try {
            while (capture.read(frame)) {
                if (frame.empty()) continue;
                try {
                    // Кадр оригинала кодируется в JPEG (≈ 50-150 KB на FullHD-кадр) —
                    // PNG для большого BGR-кадра был бы в 5-10 раз тяжелее и быстро забил
                    // бы кучу при длинных видео. Маска остаётся в PNG: бинарная маска
                    // сжимается PNG'ом до ~5 KB.
                    Image originalImage = matToImage(frame, ".jpg", 80);
                    // Маска показывается as-is (v1) — это «исходная» маска кадра.
                    // v2 — отдельный шаг: получает ту же маску и оставляет только
                    // прямолинейные компоненты, его результат уходит в детектор для
                    // выделения линий. Сама маска при этом не модифицируется.
                    Mat rawMask = maskExtractor.process(frame);
                    Image maskImage = matToImage(rawMask, ".png", 0);
                    Mat processed = detector.process(frame, rawMask);
                    Image fxImage = matToImage(processed, ".jpg", 80);

                    originalFrames.add(originalImage);
                    processedFrames.add(fxImage);
                    maskFrames.add(maskImage);

                    rawMask.release();
                    processed.release();
                } catch (Throwable t) {
                    System.err.println("[startProcessing] Frame " + frameIdx + " threw: " + t);
                    t.printStackTrace();
                    throw t;
                }
                frameIdx++;
            }
        } finally {
            capture.release();
            frame.release();
            maskExtractor.release();
            detector.release();
        }

        if (processedFrames.isEmpty()) return;

        Platform.runLater(() -> {
            originalVideoFrames  = originalFrames;
            processedVideoFrames = processedFrames;
            maskVideoFrames      = maskFrames;
            currentFps = fps;
            setControlsDisabled(false);

            // На всякий случай останавливаем предыдущий таймер ещё раз — между loadVideo
            // и этой колбэк-нагрузкой что-то могло его пересоздать.
            if (activeTimer != null) activeTimer.stop();

            // Синхронизатор Result/Mask с player1. Работает ВСЕГДА (а не только при PLAYING),
            // чтобы пауза, остановка, перемотка ползунком и доход до конца отображались
            // одинаково на всех трёх панелях. Без PLAYING-гарда: после паузы view2/view3
            // не «замораживаются» на старом кадре, а показывают реальное время плеера.
            activeTimer = new AnimationTimer() {
                @Override
                public void handle(long now) {
                    if (player1 == null || processedVideoFrames.isEmpty()) return;

                    double currentTime = player1.getCurrentTime().toSeconds();
                    int frameIndex = (int) (currentTime * currentFps);
                    if (frameIndex < 0) frameIndex = 0;
                    if (frameIndex >= processedVideoFrames.size()) {
                        frameIndex = processedVideoFrames.size() - 1;
                    }

                    if (frameIndex < originalVideoFrames.size()) {
                        view1.setImage(originalVideoFrames.get(frameIndex));
                    }
                    view2.setImage(processedVideoFrames.get(frameIndex));
                    if (frameIndex < maskVideoFrames.size()) {
                        view3.setImage(maskVideoFrames.get(frameIndex));
                    }
                }
            };
            activeTimer.start();
        });
    }

    private void setControlsDisabled(boolean disabled) {
        playButton.setDisable(disabled);
        pauseButton.setDisable(disabled);
        stopButton.setDisable(disabled);
        seek1.setDisable(disabled);
        volume1.setDisable(disabled);
    }

    /**
     * Кодирует {@code mat} в указанный формат и оборачивает в JavaFX {@link Image}.
     * Для JPEG передаётся параметр качества (0-100), для PNG он игнорируется.
     * <ul>
     *   <li>Цветной BGR-кадр FullHD в PNG ≈ 1-3 МБ, в JPEG-80 ≈ 50-150 КБ.</li>
     *   <li>Бинарная маска в PNG ≈ 5-20 КБ; в JPEG то же или больше, плюс артефакты.</li>
     * </ul>
     */
    private Image matToImage(Mat mat, String ext, int jpegQuality) {
        MatOfByte buffer = new MatOfByte();
        if (".jpg".equalsIgnoreCase(ext) || ".jpeg".equalsIgnoreCase(ext)) {
            MatOfInt params = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, jpegQuality);
            try { Imgcodecs.imencode(ext, mat, buffer, params); }
            finally { params.release(); }
        } else {
            Imgcodecs.imencode(ext, mat, buffer);
        }
        Image img;
        try { img = new Image(new ByteArrayInputStream(buffer.toArray())); }
        finally { buffer.release(); }
        return img;
    }
}