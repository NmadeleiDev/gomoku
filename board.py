from __future__ import annotations

from hashlib import md5
from typing import Optional

import numpy as np
import tabulate

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


class Board:
    empty_color = 0
    size = 20

    def __init__(
        self, position: Optional[np.ndarray] = None, move_idx=0, from_move=None
    ):
        if position is not None:
            self.position = position
        else:
            self.position = np.zeros((self.size, self.size), dtype=int)

        self.position.flags.writeable = False
        self.hash = None
        self.move_idx = move_idx
        self.from_move = from_move
        self.h_val = None

    def is_point_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.position.shape[0] and 0 <= y < self.position.shape[1]

    def is_point_empty(self, x: int, y: int) -> bool:
        return self.position[x, y] == self.empty_color

    def winner(self, criteria) -> int | None:
        winner = criteria(None, self)
        if winner != self.empty_color:
            return winner
        else:
            return None

    def compute_h_for_self(self, h):
        self.h_val = h(self)
        return self

    def get_board_after_move(self, x: int, y: int, color: int) -> Board:
        if not self.is_point_empty(x, y):
            raise ValueError("illegal move")
        result = self.position.copy()
        result.flags.writeable = True
        result[x, y] = color
        return Board(result, move_idx=self.move_idx + 1, from_move=(x, y))

    def get_point_neighbours_to_all_directions(
        self, x: int, y: int, at_distance=1
    ) -> np.ndarray:
        result = unary_step_vectors * at_distance + np.array([[x, y]], dtype=int)
        return result[
            np.apply_along_axis(lambda p: self.is_point_on_board(*p), 1, result)
        ]

    def get_center_square_points(self) -> np.ndarray:
        global center_square_points

        if center_square_points is not None:
            return center_square_points

        middle = self.position.shape[0] // 2

        middle = np.array([middle, middle], dtype=int)

        if self.position.shape[0] % 2 == 0:
            center_square_points = np.concatenate(
                [middle[np.newaxis, :], unary_step_vectors[[1, 3, 5]] + middle], axis=0
            )
        else:
            center_square_points = np.concatenate(
                [middle[np.newaxis, :], unary_step_vectors + middle], axis=0
            )

        return center_square_points

    def generate_hash(self):
        return int(md5(self.position.tobytes()).hexdigest(), 16)

    def __hash__(self):
        if self.hash is None:
            self.hash = self.generate_hash()
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return (other.position == self.position).all()

    def print_board(self, players_chars: dict[int, str]):
        board_position_copy = self.position.copy().astype(object)
        board_position_copy.flags.writeable = True

        for color, char in (players_chars | {0: "."}).items():
            board_position_copy[board_position_copy == color] = char

        print(
            tabulate.tabulate(
                board_position_copy, headers="keys", stralign="center", showindex=True
            )
        )
