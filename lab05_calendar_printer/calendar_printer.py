from typing import List

week_header = "S M T W T F S"
weekday_index = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]

#build calendar rows
def normalise_start_day(user_value: str) -> int:
    #turn user input into index where 0 is Sunday and 6 is Saturday
    #accept: 
    # - weekday strings: "sun", "sunday", "Mon", "tues", etc
    # - integer as strings: 0-6 where 0 is Sunday and 6 is Saturday
    #this design handles input robustly - keeps UX clean and main() tidy

    s = user_value.strip().lower()

    #numeric path first
    if s.isdigit():
        idx = int(s)
        if 0 <= idx <= 6:
            return idx
        raise ValueError("Numeric day index must be in range 0-6 where Sunday is 0")
    
    #text path - match by prefix
    for i, name in enumerate(weekday_index):
        if s.startswith(name):
            return i
        
    raise ValueError("Could not interpret weekday name. Use 0-6 or a weekday name like 'mon' or 'tuesday'")

def build_calendar_rows(days_in_month: int, start_day_idx: int) -> List[str]:
    #build the rows of the calendar as list of strings (weeks)
    #- each day is width = 2 and right aligned
    #- empty slots are two spaces
    #example week row: "       1  2  3  4  5  6" if month starts on Friday

    if not (1 <= days_in_month <= 31):
        raise ValueError("days_in_month must be in range 1-31")
    if not (0 <= start_day_idx <= 6):
        raise ValueError("start_day_idx must be in range 0-6 where Sunday is 0")
    
    #start first week with ' ' for days before 1st
    slots: List[str] = ['  '] * start_day_idx
    rows: List[str] = []

    for day in range(1, days_in_month + 1):
        slots.append(f"{day:>2}") #right align to 2 chars
        if len(slots) == 7:
            rows.append(" ".join(slots))
            slots = []

    #if last week isn't full, pad to 7 cols
    if slots:
        while len(slots) < 7:
            slots.append("  ")
        rows.append(" ".join(slots))

    return rows


#I/O wrapper: prompt, validate, print

def main() -> None:
    print("\nCalendar Printer (weeks start on Sunday)\n")
    print("Enter weekday as 0-6 (Sun=0)\nOR\nas a name like 'Mon' or 'Tuesday'.\n")

    #1. days in month
    while True:
        try:
            days_input = input("How many days are in the month (28-21)?\n -> ").strip()
            days = int(days_input)
            if not (1 <= days <= 31):
                raise ValueError
            break
        except ValueError:
            print("  -> Please enter whole mnumber between 1 and 31.\n")

    #2. starting weekday
    while True:
        try:
            start_input = input("What day does the month start on? (Sun=0 or name)\n -> ").strip()
            start_idx = normalise_start_day(start_input)
            break
        except ValueError as e:
            print(f"  -> {e}\n")

    #3. build + print
    rows = build_calendar_rows(days, start_idx)
    print("\n" + week_header)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()