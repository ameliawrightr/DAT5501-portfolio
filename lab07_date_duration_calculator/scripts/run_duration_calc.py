from lab07_date_duration_calculator.src.duration_calc import (
    days_since, 
    load_dates_and_calculate
)


if __name__ == "__main__":
    mode = input("Type '1' for single data or '2' to load from CSV: ")

    if mode == '1':
        date_input = input("Enter a date (YYYY-MM-DD): ")
        try:
            result = days_since(date_input)
            print(f"{abs(result)} days {'ago' if result >= 0 else 'from now'}.")
        except Exception as e:
            print(f"Error: {e}")

    elif mode == '2':
        path = "lab07_date_duration_calculator/data/random_dates.csv"
        df = load_dates_and_calculate(path)
        print("\nResults:\n", df)

    else:
        print("Invalid option selected.")