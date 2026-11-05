"""siyu."""
from numbers import Number
from math import sin, cos


class RPCalc:
    """RPCalc."""

    def __init__(self):
        """Define the entity of a stack."""
        self.stack = []

    def push(self, n):
        """push."""
        if isinstance(n, Number):
            self.stack.append(n)
        elif n == '+':
            value1 = self.stack.pop()
            value2 = self.stack.pop()
            self.stack.append(value2 + value1)
        elif n == '-':
            value1 = self.stack.pop()
            value2 = self.stack.pop()
            self.stack.append(value2 - value1)
        elif n == '*':
            value1 = self.stack.pop()
            value2 = self.stack.pop()
            self.stack.append(value2 * value1)
        elif n == '/':
            value1 = self.stack.pop()
            value2 = self.stack.pop()
            self.stack.append(value2 / value1)
        elif n == "sin":
            value1 = self.stack.pop()
            self.stack.append(sin(value1))
        elif n == "cos":
            value1 = self.stack.pop()
            self.stack.append(cos(value1))

    def pop(self):
        """pop."""
        return self.stack.pop()

    def peek(self):
        """peek."""
        return self.stack[-1]

    def __len__(self):
        """len."""
        return len(self.stack)
