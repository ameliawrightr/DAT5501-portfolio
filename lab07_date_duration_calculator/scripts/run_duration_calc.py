from lab07_date_duration_calculator.src.duration_calc import days_since


if __name__ == "__main__":
    date_input = input("Enter a date (YYYY-MM-DD): ")
    try:
        result = days_since(date_input)
        print(f"{abs(result)} days {'ago' if result >= 0 else 'from now'}.")
    except Exception as e:
        print(f"Error: {e}")