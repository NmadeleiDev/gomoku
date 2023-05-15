from hashlib import md5
from typing import Callable, Optional
import joblib
import os

import numpy as np
import tabulate
from numba import jit_module, jit

from exceptions import IllegalMove

unary_step_vectors = np.array(
    [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [1, -1],
        [-1, 1],
        [-1, -1],
    ],
    dtype=int,
)

center_square_points = None
empty_color = 0

os.environ['PYTHONHASHSEED'] = '0'


def get_center_square_points(position: np.ndarray) -> np.ndarray:
    global center_square_points

    if center_square_points is not None:
        return center_square_points

    middle = position.shape[0] // 2

    middle = np.array([middle, middle], dtype=int)

    if position.shape[0] % 2 == 0:
        center_square_points = np.concatenate(
            [middle[np.newaxis, :], unary_step_vectors[[1, 3, 5]] + middle], axis=0
        )
    else:
        center_square_points = np.concatenate(
            [middle[np.newaxis, :], unary_step_vectors + middle], axis=0
        )

    return center_square_points

def is_point_on_board(position: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < position.shape[0] and 0 <= y < position.shape[1]

def is_point_empty(position, x: int, y: int) -> bool:
    return position[x, y] == empty_color

def get_point_neighbours_coords_to_all_directions(
    position: np.ndarray, x: int, y: int, at_distance=1, filter_by_on_board=True
) -> np.ndarray:
    result = unary_step_vectors * at_distance + np.array([[x, y]], dtype=int)
    return (
        result[np.apply_along_axis(lambda p: is_point_on_board(position, *p), 1, result)]
        if filter_by_on_board
        else result
    )

def perform_inplace_capture_if_possible(position: np.ndarray, from_move: tuple[int, int], captures: dict[int, int]) -> np.ndarray:
    x, y = from_move
    move_color = position[x, y]
    capture_pattern = np.array(
        [move_color, -move_color, -move_color, move_color], dtype=int
    )

    if move_color not in captures:
        captures[move_color] = 0

    for idx in np.stack(
        [
            get_point_neighbours_coords_to_all_directions(
                position, x, y, d, filter_by_on_board=False
            )
            for d in range(4)
        ],
        axis=1,
    ):
        idx_t = idx.T
        try:
            if np.all(position[idx_t[0], idx_t[1]] == capture_pattern):
                captures[move_color] += 1
                position[idx_t[0][1:-1], idx_t[1][1:-1]] = empty_color
        except IndexError:
            continue

    return position

def get_board_after_move(position: np.ndarray, x: int, y: int, color: int, captures: dict[int, int] = {}) -> np.ndarray:
    if not is_point_empty(position, x, y):
        raise IllegalMove("illegal move")
    result = position.copy()
    result[x, y] = color
    return perform_inplace_capture_if_possible(result, (x, y), captures)

def get_move_between_boards(board_1: np.ndarray, board_2: np.ndarray) -> tuple[int, int]:
    nonequal_points = np.argwhere(np.logical_and(board_1 != board_2, board_2 != empty_color))
    if len(nonequal_points) != 1:
        raise IllegalMove(f'len(nonequal_points) != 1: {len(nonequal_points)}: {nonequal_points}\n'
                          f'{board_str(board_1, {0: ".", 1: "X", -1: "O"})}\n'
                          f'{board_str(board_2, {0: ".", 1: "X", -1: "O"})}')
    return nonequal_points[0]

@jit(forceobj=True)
def get_position_hash(position: np.ndarray) -> int:
    return int(md5(position.tobytes()).hexdigest(), 16)
    # return hash(position)

def board_str(position: np.ndarray, players_chars: dict[int, str]) -> str:
    board_position_copy = position.copy().astype(object)

    for color, char in (players_chars | {0: "."}).items():
        board_position_copy[board_position_copy == color] = char

    return tabulate.tabulate(
            board_position_copy, headers="keys", stralign="center", showindex=True
        )

jit_module(nopython=True, error_model="numpy")
