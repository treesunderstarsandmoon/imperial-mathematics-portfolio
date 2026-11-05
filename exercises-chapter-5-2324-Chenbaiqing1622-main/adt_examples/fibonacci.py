"""ghcfhchguv."""


def fib(n):
    """Return the n-th Fibonacci number."""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-2) + fib(n-1)


class Fib:
    """Iterator for Fibonacci numbers."""

    def __init__(self):
        """Initialize the starting values."""
        self.prev, self.curr = 1, 0

    def __iter__(self):
        """Return the iterator itself."""
        return self

    def __next__(self):
        """Return the next Fibonacci number."""
        if self.curr == 0:  # Adjust the first step to produce 1
            self.curr = 1
            return 1
        # Calculate the next Fibonacci number
        next_value = self.prev + self.curr
        # Update prev and curr to the next values in the sequence
        self.prev, self.curr = self.curr, next_value
        # Return the previous value, which is the current Fibonacci number
        return self.curr
