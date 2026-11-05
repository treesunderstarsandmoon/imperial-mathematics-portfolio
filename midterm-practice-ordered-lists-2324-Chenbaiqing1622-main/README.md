# Principles of Programming Midterm Test

## Instructions

* Time allowed: 40 minutes.
* This paper has one question.

1. Ensure your Principles of Programming venv is active.
2. Clone this repository.
3. Complete the programming tasks. Tests are provided for all parts except
   question 1(a)(i), which is tested when you push, and for the style marks. You should run the tests and check flake8 for every question part as you go.
4. Commit and push as you go. Even if you have network issues which prevent you
   from pushing, you can still make commits locally and it is essential that
   you do. **Do not** rely on one big commit and push at the
   end of the test. Only commits timestamped before the end of the test
   will count.
5. Ensure that your code and any code provided in the question passes Flake8 and
   otherwise conforms to good programming style as we have learned in this
   course. There is no need to write any docstrings in this test.

The tests and autograder are a guide, the test will be manually marked in order
to ensure that your code doesn't produce the right output for the wrong reasons,
and in order to allocate the style marks.

Where a question indicates the arguments a method takes, these are in addition
to the the `self` argument that all instance methods take.

## Question 1 Ordered list

In this question you will implement a list data structure which always stores
its contents in ascending order by value. Because this will frequently involve
inserting values into the middle of the list, we will base the implementation
on a linked list.

As an illustrative example, the following code:

```python
from lists import OrderedList
ol = OrderedList()
ol.insert(10)
ol.insert(0)
ol.insert(5)
print(ol)
```
should print:
```
[0, 5, 10]
```

### 1 (a) 

Your test repository contains a package `lists`. This contains a module
`linked_list` which in turn contains the classes `Link` and `LinkIterator`
which were introduced in chapter 5 of the course text.

#### (i) (1 mark)

Make the package installable. You will aid your own
completion of the test by installing this package in editable mode.

#### (ii) (1 mark)

Create a new module `ordered_list` in the `lists` package.

### 1 (b)

#### (i) (2 marks)
Create a class `lists.ordered_list.OrderedList` whose constructor takes no
arguments other than `self`. The constructor should set the attribute `_length`
to `0` and the attribute `_first` to `None`. `_length` will record the length of
the list, and `_first` will eventually contain the
first link in the linked list which will store the values in the list.

#### (ii) (1 mark)
Modify your package so that the following Python code will work:

```python
from lists import OrderedList
```

#### (iii) (1 mark)
Implement the
[`__len__`](https://docs.python.org/3/reference/datamodel.html#object.__len__)
special method of `OrderedList` so that it returns the current length of the list.

#### (iv) (2 marks)

Implement the [`__iter__`](https://docs.python.org/3/reference/datamodel.html#object.__iter__) special method of `OrderedList` so that the list is iterable. You can
simply return a `lists.linked_list.ListIterator` to which you have passed `self._first`.

#### (v) (1 mark)

Implement the [`__str__`](https://docs.python.org/3/reference/datamodel.html#object.__str__) special method of `OrderedList` so that it returns an identical result
to the string representation of an equivalent `list`. You can just convert
`self` to a `list` and convert the resulting `list` to a `str`.

### 1 (c) 

#### (i) (3 marks)

Implement a new method of `OrderedList` called `pop`. Pop should remove and return the lowest value in the list. If the list is empty then the method should raise `IndexError` with the message "pop from empty list".

#### (ii) (4 marks)

Implement a new method of `OrderedList` called `insert`. `insert` should take, in addition to `self`, an argument called `value`. Your code should insert a new link with the given value into the linked list which starts with `self._first`. The insertion must ensure that the items in the list are always in increasing order.

If the list is currently empty or if the value to be inserted is smaller than any value in the list, then the new link must be inserted at the start of the list and `self._first` re-assigned appropriately. In any other case, your code will need to loop through the linked list to insert the new link at the appropriate point.

### 1 (d) 

#### (i) (2 marks)
Ensure that the code for this exercise passes Flake8.

#### (ii) (2 marks)
Ensure that your code otherwise conforms to good programming style as we have
learned in this course.

There is no need to write any docstrings in this test.
