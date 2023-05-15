from functools import cache
from itertools import product
from typing import Callable
from cachetools import cached

import numpy as np
from numba import jit

board_size=19
empty_color=0
power_foundation = board_size

@cache
def get_board_sliding_indices(line_len_to_analyze=5):
    window_path_len = board_size - line_len_to_analyze + 1

    straight_line_indices_0 = [
        np.arange(line_len_to_analyze) + i for i in range(window_path_len)
    ] * board_size
    straight_line_indices_1 = np.concatenate(
        [np.ones((window_path_len,)) * i for i in range(board_size)]
    ).astype(int)[:, np.newaxis]

    diags = [np.arange(line_len_to_analyze) + i for i in range(window_path_len)]
    inv_diags = [
        np.flip(np.arange(line_len_to_analyze)) + i for i in range(window_path_len)
    ]

    diag_indices_0, diag_indices_1 = [
        list(x) for x in zip(*list(product(diags, diags + inv_diags)))
    ]

    return (
        straight_line_indices_0,
        straight_line_indices_1,
        diag_indices_0,
        diag_indices_1,
    )

def get_unique_lines_from_board(position: np.ndarray, line_len_to_analyze=5):
    (
        straight_line_indices_0,
        straight_line_indices_1,
        diag_indices_0,
        diag_indices_1,
    ) = get_board_sliding_indices(line_len_to_analyze)

    lines = np.concatenate(
        [
            position[straight_line_indices_0, straight_line_indices_1],
            position[straight_line_indices_1, straight_line_indices_0],
            position[diag_indices_0, diag_indices_1],
        ]
    )

    unique_lines, unique_counts = np.unique(lines, axis=0, return_counts=True)
    return unique_lines, unique_counts


@cached(cache={}, key=lambda h_fn, whos_move, board, board_hash, line_len_to_analyze=5: (whos_move, board_hash))
@jit(nopython=True)
def apply_scalar_heuristic(
    h_fn: Callable[[int, int], float],
    whos_move: int | None,
    board: np.ndarray,
    board_hash: int,
    line_len_to_analyze=5,
):
    unique_lines, unique_counts = get_unique_lines_from_board(
        board, line_len_to_analyze=line_len_to_analyze
    )

    result = np.sum(
        [
            h_fn(whos_move, tuple(x)) * count
            for x, count in zip(unique_lines, unique_counts)
        ]
    )
    return result

@cache
@jit(nopython=True)
def sum_for_homogeneous_line(color: int, line: tuple) -> float:
    line = np.array(line, dtype=int)

    unique_colors = np.unique(line)
    if len(unique_colors) == 1 and unique_colors[0] == empty_color:
        return 0

    if len(unique_colors) == 3 or (
        len(unique_colors) == 2 and empty_color not in unique_colors
    ):
        return 0

    is_this_color_line = color in unique_colors

    count = np.sum(line != empty_color)

    score = (power_foundation**count) if count < len(line) else np.inf

    return score * (1 if is_this_color_line else -1)


@cache
@jit(nopython=True)
def is_free_three(color: int, line: tuple) -> float:
    """
    :param line: tuple длины 6
    :param color: игнорируется
    :param whos_move: цвет, free three которого мы ищем
    :return: 1 если free three есть, 0 если нет
    """
    whos_move = -color
    if not (line[0] == empty_color and line[-1] == empty_color):
        return 0

    n_stones = 0
    for i in range(1, 5):
        if line[i] == whos_move:
            n_stones += 1
        elif line[i] == empty_color:
            pass
        else:
            return 0

    if n_stones == 3:
        if line[1] == whos_move and line[-2] == whos_move:
            return 1
        else:
            return 0.5
    else:
        return 0
    
