def add(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Must be numbers")
    return a + b

a=2
b=3

print(add(a, b))