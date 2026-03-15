package sarinxo.edu.cvproject.config;

import lombok.extern.slf4j.Slf4j;
import nu.pattern.OpenCV;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;

@Slf4j
@Configuration
public class OpenCvConfiguration {

    @Bean
    public ApplicationRunner initCv() {
        return args -> {
            log.info("CV configuration initializing");
            ClassPathResource resource = new ClassPathResource("lib/opencv_java4100.dll");
            System.load(resource.getFile().getAbsolutePath());
            OpenCV.loadShared();
            log.info("CV configuration initialized");
        };
    }

}
