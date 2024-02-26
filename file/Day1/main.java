enum Cartypes {
    SUV, HATCHBACK, SEDAN, CONVERTIBLE
}

public class EnumCar {
    public static void main(String[] args) {
        Cartypes car1 = Cartypes.SEDAN;

        System.out.println("Value for car1: " + car1);
        System.out.println("Value for car1 ordinal: " + car1.ordinal());
        System.out.println("For loop:: ");

        for (Cartypes c : Cartypes.values()) {
            System.out.println(c + " = " + c.ordinal());
        }
    }
}
