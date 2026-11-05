"""gugg."""


class Deque:
    """Deque."""

    def __init__(self, n):
        """Return constructor."""
        self.size = n
        self.deque = [None] * n
        self.rightpointer = 0
        self.leftpointer = 0

    def append(self, x):
        """append."""
        if self.deque[self.rightpointer] is None:
            self.deque[self.rightpointer] = x
            self.rightpointer = (self.rightpointer + 1) % self.size
        else:
            return NotImplemented

    def appendleft(self, x):
        """appendleft."""
        if self.deque[self.leftpointer - 1] is None:
            self.deque[self.leftpointer - 1] = x
            self.leftpointer = (self.leftpointer - 1) % self.size
        else:
            return NotImplemented

    def pop(self):
        """pop."""
        pop_value = self.deque[self.rightpointer - 1]
        self.deque[self.rightpointer - 1] = None
        self.rightpointer = (self.rightpointer - 1) % self.size
        return pop_value

    def popleft(self):
        """popleft."""
        pop_value = self.deque[self.leftpointer]
        self.deque[self.leftpointer] = None
        self.leftpointer = (self.leftpointer + 1) % self.size
        return pop_value

    def peek(self):
        """peek."""
        return self.deque[self.rightpointer - 1]

    def peekleft(self):
        """peekleft."""
        return self.deque[self.leftpointer]

    def __len__(self):
        """len."""
        count = 0
        for i in self.deque:
            if i is not None:
                count = count + 1
        return count

    def __iter__(self):
        """Make iter."""
        return DequeIterator(self)


class DequeIterator:
    """Iter."""

    def __init__(self, deque):
        """kskkw."""
        self._deque = deque
        self._index = deque.leftpointer
        self._count = 0  # A counter to track the number of elements returned

    def __next__(self):
        """kskd."""
        if self._count >= len(self._deque):
            raise StopIteration
        result = self._deque.deque[self._index]
        self._index = (self._index + 1) % self._deque.size
        self._count += 1
        return result
