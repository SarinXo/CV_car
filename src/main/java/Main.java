import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        List<Integer> integers = new ArrayList<>(List.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 0));

        integers.removeIf(i -> i % 2 == 0);

        integers.forEach(System.out::println);


    }
}
