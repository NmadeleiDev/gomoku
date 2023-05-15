from __future__ import annotations

from hashlib import md5
from typing import Callable, Optional

import joblib
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
    size = 19

    def __init__(
        self,
        position: Optional[np.ndarray] = None,
        move_idx: int = 0,
        from_move: int = None,
        last_move_color: int = None,
        captures: dict[int, int] = None,
        free_threes_count: dict[int, int] = None,
        players_chars: dict[int, str] = None,
    ):
        if position is not None:
            self.position = position
        else:
            self.position = np.zeros((self.size, self.size), dtype=int)

        self.position.flags.writeable = False
        self.hash = None
        self.move_idx = move_idx
        self.from_move = from_move
        self.last_move_color = last_move_color
        self.h_val = None
        self.captures = captures.copy() if captures is not None else {}
        self.free_threes_count = (
            free_threes_count.copy() if free_threes_count is not None else {}
        )
        self.players_chars = players_chars

    def is_point_on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.position.shape[0] and 0 <= y < self.position.shape[1]

    def is_point_empty(self, x: int, y: int) -> bool:
        return self.position[x, y] == self.empty_color

    def winner(self, criteria) -> int | None:
        for color, n_captures in self.captures.items():
            if n_captures >= 5:
                return color

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
        return Board(
            result,
            move_idx=self.move_idx + 1,
            from_move=(x, y),
            last_move_color=color,
            captures=self.captures,
            free_threes_count=self.free_threes_count,
        ).perform_inplace_capture_if_possible()

    def perform_inplace_capture_if_possible(self):
        x, y = self.from_move
        move_color = self.position[x, y]
        capture_pattern = np.array(
            [move_color, -move_color, -move_color, move_color], dtype=int
        )

        if move_color not in self.captures:
            self.captures[move_color] = 0

        self.position.flags.writeable = True

        for idx in np.stack(
            [
                self.get_point_neighbours_coords_to_all_directions(
                    x, y, d, filter_by_on_board=False
                )
                for d in range(4)
            ],
            axis=1,
        ):
            idx_t = idx.T
            try:
                if np.all(self.position[idx_t[0], idx_t[1]] == capture_pattern):
                    self.captures[move_color] += 1
                    self.position[idx_t[0][1:-1], idx_t[1][1:-1]] = self.empty_color
            except IndexError:
                continue

        self.position.flags.writeable = False
        return self

    def update_double_free_three_count_and_check_if_violated(
        self, free_three_counter: Callable[[int, Board], int]
    ) -> bool:
        if self.move_idx < 8:
            return False

        if self.last_move_color not in self.free_threes_count:
            self.free_threes_count[self.last_move_color] = 0

        new_count = free_three_counter(self.last_move_color, self)

        is_ok = new_count - self.free_threes_count[self.last_move_color] > 1

        self.free_threes_count[self.last_move_color] = new_count

        return is_ok

    def get_point_neighbours_coords_to_all_directions(
        self, x: int, y: int, at_distance=1, filter_by_on_board=True
    ) -> np.ndarray:
        result = unary_step_vectors * at_distance + np.array([[x, y]], dtype=int)
        return (
            result[np.apply_along_axis(lambda p: self.is_point_on_board(*p), 1, result)]
            if filter_by_on_board
            else result
        )

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

    def __str__(self) -> str:
        board_position_copy = self.position.copy().astype(object)
        board_position_copy.flags.writeable = True

        for color, char in (self.players_chars | {0: "."}).items():
            board_position_copy[board_position_copy == color] = char

        return tabulate.tabulate(
            board_position_copy, headers="keys", stralign="center", showindex=True
        )

    def dump(self, suffix: str = ""):
        joblib.dump(self, f"./logs/board_{suffix}.joblib")

    @staticmethod
    def load(checkpoint_path: str) -> Board:
        return joblib.load(checkpoint_path)
