from lab05_calendar_printer.calendar_printer import build_calendar_rows

def test_basic_layout():
    #30 day month starting on Wednesday (Sun=0 -> Wed=3)
    rows = build_calendar_rows(30,3)
    assert rows[0].startswith("      1") #3 blanks -> 6 spaces
    #should create either 5 or 6 rows depending on layout
    assert len(rows) == 5
    #last printed number should be 30 somwhere in the grid
    assert any("30" in r for r in rows)
    