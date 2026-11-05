import pytest
import numpy as np
try:
    from lists.ordered_list import OrderedList
    from lists.linked_list import Link, LinkIterator
except ImportError:
    pass

@pytest.fixture
def example_list():
    ol = OrderedList()
    links = Link(-1, Link(0, Link(2.5, Link(50))))
    ol._first = links
    ol._length = 4
    return ol


@pytest.mark.q1_aii
def test_import_ordered_list():
    import lists.ordered_list


@pytest.mark.q1_bi
def test_instantiate_ordered_list():
    ol = OrderedList()
    assert ol._first is None
    assert ol._length == 0


@pytest.mark.q1_bii
def test_direct_import_ordered_list():
    from lists import OrderedList


@pytest.mark.q1_biii
@pytest.mark.parametrize("value", range(5))
def test_list_length(value):
    ol = OrderedList()
    ol._length = value
    assert len(ol) == value

@pytest.mark.q1_biv
def test_list_iter_type(example_list):
    li = iter(example_list)
    assert isinstance(li, LinkIterator), \
        f"List iteration returns a {type(li).__name__} expected a LinkIterator"

@pytest.mark.q1_biv
def test_list_iter_value(example_list):
    for item, expected in zip(example_list, [-1, 0, 2.5,50]):
        assert item == expected


@pytest.mark.q1_biv
def test_list_iter_to_list(example_list):
    assert list(example_list) == [-1, 0, 2.5, 50]


@pytest.mark.q1_bv
def test_list_str(example_list):
    assert str(example_list) == "[-1, 0, 2.5, 50]"


@pytest.mark.q1_ci
def test_list_pop(example_list):
    for i, val in zip(range(3, -1, -1), [-1, 0, 2.5, 50]):
        assert example_list.pop() == val
        assert len(example_list) == i


@pytest.mark.q1_ci
def test_list_pop_empty():
    ol = OrderedList()
    with pytest.raises(IndexError, match="pop from empty list"):
        ol.pop()


@pytest.mark.q1_cii
def test_insert():
    ol = OrderedList()
    assert len(ol) == 0
    for i, value in enumerate([2.5, -1, 50, 0], start=1):
        ol.insert(value)
        assert len(ol) == i

    link = ol._first
    assert link.value == -1
    for value in [0, 2.5, 50]:
        link = link.next
        assert link.value == value, \
            f"Incorrect value in list. Got {link.value}, expected {value}."
    assert link.next is None, \
        f"Expected None at end of list, got {link.next}"
