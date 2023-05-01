from functools import cache, partial
from itertools import product
from typing import Callable

import numpy as np
from cachetools import cached

from board import Board

power_foundation = Board.size


@cache
def get_board_sliding_indices(line_len_to_analyze=5):
    window_path_len = Board.size - line_len_to_analyze + 1

    straight_line_indices_0 = [
        np.arange(line_len_to_analyze) + i for i in range(window_path_len)
    ] * Board.size
    straight_line_indices_1 = np.concatenate(
        [np.ones((window_path_len,)) * i for i in range(Board.size)]
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


def hamming_similarity(l1: np.ndarray, l2: np.ndarray):
    return np.sum(l1 == l2)


@cache
def score_for_line_hamming(color: int, line: tuple) -> float:
    line = np.array(line, dtype=int)
    result = 0

    for color_in_line in np.unique(line):
        if color_in_line == Board.empty_color:
            continue
        win_line_for_color = np.empty((len(line),))
        win_line_for_color.fill(color_in_line)
        ham_sim = hamming_similarity(
            line, win_line_for_color
        )  # TODO различие с пустым местом и камнем оппонента учитывать по-разному
        score = (power_foundation**ham_sim) if ham_sim < len(line) else np.inf
        result += score * (1 if color_in_line == color else -1)

    return result


@cache
def score_for_line_count_with_move(color: int, line: tuple, whos_move: int) -> float:
    line = np.array(line, dtype=int)

    unique_colors = np.unique(line)
    if len(unique_colors) == 1 and unique_colors[0] == Board.empty_color:
        return 0

    if len(unique_colors) == 3 or (
        len(unique_colors) == 2 and Board.empty_color not in unique_colors
    ):
        return 0

    is_this_color_line = color in unique_colors

    count = np.sum(line != Board.empty_color)

    score = None
    if (is_this_color_line and color == whos_move) or (
        not is_this_color_line and color != whos_move
    ):
        if (
            count == 3
            and line[0] == Board.empty_color
            and line[-1] == Board.empty_color
        ):
            score = np.inf
        elif count == 4 and (
            line[0] == Board.empty_color or line[-1] == Board.empty_color
        ):
            score = np.inf
        elif count == 5:
            score = np.inf
    else:
        if count == 5 or (count == 4 and Board.empty_color in line):
            score = np.inf

    if score is None:
        score = (power_foundation**count) if count < len(line) else np.inf

    return score * (1 if is_this_color_line else -1)


@cache
def score_for_line_count(color: int, line: tuple, whos_move: int) -> float:
    line = np.array(line, dtype=int)

    unique_colors = np.unique(line)
    if len(unique_colors) == 1 and unique_colors[0] == Board.empty_color:
        return 0

    if len(unique_colors) == 3 or (
        len(unique_colors) == 2 and Board.empty_color not in unique_colors
    ):
        return 0

    is_this_color_line = color in unique_colors

    count = np.sum(line != Board.empty_color)

    score = (power_foundation**count) if count < len(line) else np.inf

    return score * (1 if is_this_color_line else -1)


@cached(cache={}, key=lambda color, line, whos_move: (line,))
def whos_win(color: None, line: tuple, whos_move: None) -> int:
    line = np.array(line, dtype=int)

    unique_colors = np.unique(line)
    if len(unique_colors) == 1 and unique_colors[0] != Board.empty_color:
        return unique_colors[0]
    else:
        return 0


@cached(cache={}, key=lambda color, line, whos_move: (line, whos_move))
def is_free_three(color: None, line: tuple, whos_move: int) -> float:
    """
    :param line: tuple длины 6
    :param color: игнорируется
    :param whos_move: цвет, free three которого мы ищем
    :return: 1 если free three есть, 0 если нет
    """
    if not (line[0] == Board.empty_color and line[-1] == Board.empty_color):
        return 0

    n_stones = 0
    for i in range(1, 5):
        if line[i] == whos_move:
            n_stones += 1
        elif line[i] == Board.empty_color:
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


def get_unique_lines_from_board(board: Board, line_len_to_analyze=5):
    (
        straight_line_indices_0,
        straight_line_indices_1,
        diag_indices_0,
        diag_indices_1,
    ) = get_board_sliding_indices(line_len_to_analyze)

    position = board.position

    lines = np.concatenate(
        [
            position[straight_line_indices_0, straight_line_indices_1],
            position[straight_line_indices_1, straight_line_indices_0],
            position[diag_indices_0, diag_indices_1],
            # position[diag_indices_1, diag_indices_0]
        ]
    )

    unique_lines, unique_counts = np.unique(lines, axis=0, return_counts=True)
    return unique_lines, unique_counts


@cache
def apply_scalar_heuristic(
    h_fn: Callable[[tuple, int], float],
    whos_move: int | None,
    board: Board,
    line_len_to_analyze=5,
):
    unique_lines, unique_counts = get_unique_lines_from_board(
        board, line_len_to_analyze=line_len_to_analyze
    )

    result = sum(
        [
            h_fn(tuple(x), whos_move) * count
            for x, count in zip(unique_lines, unique_counts)
        ]
    )
    return result


class Heuristics:
    hamming = score_for_line_hamming
    count = score_for_line_count
    count_with_move = score_for_line_count_with_move
    bin = whos_win
    free_three = is_free_three


def build_heuristic(
    color: int | None,
    scorer_fn: Callable[[int, tuple, int], int],
    line_len_to_analyze=5,
) -> Callable[[int | None, Board], float]:
    scorer_fn = partial(scorer_fn, color)

    return partial(
        apply_scalar_heuristic, scorer_fn, line_len_to_analyze=line_len_to_analyze
    )
