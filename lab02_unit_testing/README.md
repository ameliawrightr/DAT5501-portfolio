# DAT5501 – Unit Testing Lab

## Overview  
This lab demonstrates **unit testing** and **Git branching/merging**.  
I implemented a simple `add(a, b)` function and validated it with pytest.

## Function Design  
- Adds two numbers (`int` or `float`).  
- Raises `TypeError` for invalid inputs (`str`, `None`, `bool`, etc.).  
- Returns `int` if both inputs are ints, otherwise `float`.

### Examples
```python
add(2, 3)     # 5
add(2.0, 3)   # 5.0
add(True, 1)  # raises TypeError
