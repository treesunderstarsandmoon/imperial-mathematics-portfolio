"""numpy."""

import numpy as np
from matplotlib import pyplot
from scipy.signal import convolve2d

glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])

blinker = np.array([
 [0, 0, 0],
 [1, 1, 1],
 [0, 0, 0]]
)

glider_gun = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0]
])


class Game:  # noqa D101
    def __init__(self, size):
        """Play game."""
        self. board = np.zeros((size, size))

    def play(self):  # noqa D102
        print("Playing life. Press ctrl + c to stop.")
        pyplot.ion()
        while True:
            self.move()
            self.show()
            pyplot.pause(0.0000005)

    def move(self):
        """move."""
        stencil = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbourcount = convolve2d(self.board, stencil, mode='same')

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                self.board[i, j] = 1 if (neighbourcount[i, j] == 3
                                         or (neighbourcount[i, j] == 2
                                         and self.board[i, j])) else 0

    def __setitem__(self, key, value):  # noqa D102
        self.board[key] = value

    def show(self):  # noqa D102
        pyplot.clf()  # noqa E999
        pyplot.matshow(self.board, fignum=0, cmap='binary')
        pyplot.show()

    def insert(self, pattern, center_coords):
        """insert."""
        center_row, center_col = center_coords
        pattern_grid = pattern.grid
        rows, cols = pattern_grid.shape

        start_row = center_row - rows // 2
        start_col = center_col - cols // 2

        end_row = start_row + rows
        end_col = start_col + cols

        # Adjust indices in case the pattern goes out of the board boundaries
        start_row = max(start_row, 0)
        start_col = max(start_col, 0)
        end_row = min(end_row, self.board.shape[0])
        end_col = min(end_col, self.board.shape[1])

        # Calculate the size of the slice we're actually able to insert,
        # in case we had to adjust the starting and ending indices
        actual_rows = end_row - start_row
        actual_cols = end_col - start_col

        # Place the pattern onto the board slice, ensuring we only insert
        # the part of the pattern that fits within the adjusted slice size
        self.board[start_row:end_row, start_col:end_col] = pattern_grid[
            :actual_rows, :actual_cols]


class Pattern:
    """pattern."""

    def __init__(self, array):
        """Construct."""
        self.grid = array

    def flip_vertical(self):
        """Reverse the pattern."""
        return Pattern(self.grid[::-1, :])

    def flip_horizontal(self):
        """Reverse the pattern."""
        return Pattern(self.grid[:, ::-1])

    def flip_diag(self):
        """Transpose."""
        return Pattern(np.transpose(self.grid))

    def rotate(self, n):
        """Rotate."""
        return Pattern(np.rot90(self.grid, n))
