from typing import List

week_header = "S  M  T  W  T  F  S"
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

    #first visible day in first week must follow exactly 2*start_day_idx spaces
    # i.e, no extra padding before '1'
    #subsequent printed days use width = 2 with single space separator

    if not (1 <= days_in_month <= 31):
        raise ValueError("days_in_month must be in range 1-31")
    if not (0 <= start_day_idx <= 6):
        raise ValueError("start_day_idx must be in range 0-6 where Sunday is 0")
    
    #fill 6x7 grid with day numbers or None
    grid: List[List[int | None]] = [[None for _ in range(7)] for _ in range(6)]
    pos = start_day_idx
    r = 0
    for day in range(1, days_in_month + 1):
        grid[r][pos] = day
        pos += 1
        if pos == 7:
            pos = 0
            r += 1
    
    #start first week with ' ' for days before 1st
    rows: List[str] = []
    
    #format first non empty week to satisfy "exact 6 spaces" test
    first_row_idx = 0
    #find first week that has any day
    while first_row_idx < 6 and all(c is None for c in grid[first_row_idx]):
        first_row_idx += 1

    for i in range(first_row_idx, 6):
        week = grid[i]
        if all(c is None for c in week):
            break #stop at first completely empty week at end

        if i == first_row_idx:
            #special first line
            #prefix: 2 spaces per leading blank w/o separators
            lead = 0 
            while lead < 7 and week[lead] is None: 
                lead += 1
            line = "  " * lead #e.g., start_day_idx=3 -> "      " (6 spaces)

            if lead < 7:
                #first visible day printed with no internal left pad
                line += str(week[lead])
                col = lead + 1
                #remaining cols: single space separator + width = 2 cells
                while col < 7:
                    if week[col] is None:
                        line += " " + "  "
                    else:
                        line += " " + f"{week[col]:>2}"
                    col += 1
                rows.append(line)
        else:
            #normal lines: koin 7 columns with single space, width=2 for numbers
            cells = [(f"{d:>2}" if d is not None else "  ") for d in week]
            rows.append(" ".join(cells))

    return rows 



#I/O wrapper: prompt, validate, print

def main() -> None:
    print("\nCalendar Printer (weeks start on Sunday)\n")
    print("Enter weekday as 0-6 (Sun=0)\nOR\nas a name like 'Mon' or 'Tuesday'.\n")

    #1. days in month
    while True:
        try:
            days_input = input("How many days are in the month (28-31)?\n -> ").strip()
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