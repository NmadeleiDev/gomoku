import numpy as np

from board import Board


def first_non_equal_element_coords(position_1: Board, position_2: Board):
    return np.argwhere(np.not_equal(position_1.position, position_2.position))[0]
