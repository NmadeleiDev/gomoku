from functools import cache, partial
from itertools import product
from typing import Callable, Literal

import numpy as np

from board import Board

winning_line_len = 5
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
        win_line_for_color = np.empty((winning_line_len,))
        win_line_for_color.fill(color_in_line)
        ham_sim = hamming_similarity(
            line, win_line_for_color
        )  # TODO различие с пустым местом и камнем оппонента учитывать по-разному
        score = (power_foundation**ham_sim) if ham_sim < winning_line_len else np.inf
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
        score = (power_foundation**count) if count < winning_line_len else np.inf

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

    score = (power_foundation**count) if count < winning_line_len else np.inf

    return score * (1 if is_this_color_line else -1)


def whos_win(line: tuple, whos_move: int) -> int:
    line = np.array(line, dtype=int)

    unique_colors = np.unique(line)
    if len(unique_colors) == 1 and unique_colors[0] != Board.empty_color:
        return unique_colors[0]
    else:
        return 0


@cache
def apply_heuristic(h_fn, whos_move: int, board: Board):
    (
        straight_line_indices_0,
        straight_line_indices_1,
        diag_indices_0,
        diag_indices_1,
    ) = get_board_sliding_indices()

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

    result = sum(
        [
            h_fn(tuple(x), whos_move) * count
            for x, count in zip(unique_lines, unique_counts)
        ]
    )
    return result


def build_heuristic(
    color: int,
    scorer_type: Literal["hamming", "count", "count_with_move", "bin"] = "count",
) -> Callable[[int, Board], np.float]:
    global straight_line_indices_0, straight_line_indices_1, diag_indices_0, diag_indices_1

    if scorer_type == "count":
        scorer_fn = partial(score_for_line_count, color)
    elif scorer_type == "count_with_move":
        scorer_fn = partial(score_for_line_count_with_move, color)
    elif scorer_type == "hamming":
        scorer_fn = partial(score_for_line_hamming, color)
    elif scorer_type == "bin":
        scorer_fn = whos_win
    else:
        raise ValueError(f"unknown scorer_type: {scorer_type}")

    return partial(apply_heuristic, scorer_fn)
