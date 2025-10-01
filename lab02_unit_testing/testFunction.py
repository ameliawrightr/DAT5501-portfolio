import pytest 
from function import add

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_negative_numbers():
    assert add(-2, -3) == -5

def test_add_zero():
    assert add(0, 5) == 5
    assert add(5, 0) == 5

def test_add_invalid_type():
    with pytest.raises(TypeError):
        add(2, "3")
    with pytest.raises(TypeError):
        add("2", 3)
    with pytest.raises(TypeError):
        add("2", "3")