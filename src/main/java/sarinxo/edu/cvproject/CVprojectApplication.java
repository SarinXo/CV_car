package sarinxo.edu.cvproject;

import javafx.application.Application;
import javafx.stage.Stage;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.context.ConfigurableApplicationContext;
import sarinxo.edu.cvproject.view.DualVideoPlayer;

@Slf4j
@SpringBootApplication
public class CVprojectApplication extends Application {

    private static ConfigurableApplicationContext context;

    public static void main(String[] args) {
        context = new SpringApplicationBuilder(CVprojectApplication.class)
                .web(WebApplicationType.NONE)
                .run(args);

        launch(args);
    }

    /**
     * Запуск UI
     */
    @Override
    public void start(Stage stage) {
        DualVideoPlayer player = new DualVideoPlayer();
        player.start(stage);
    }

    @Override
    public void stop() {
        context.stop();
    }
}
