from concurrent.futures import ProcessPoolExecutor as Pool
from functools import cache, partial

import numpy as np
from cachetools import cached

from board import Board
from heuristics.sliding import build_heuristic
from player.base import Player
from player.utils import first_non_equal_element_coords


class AIPlayer(Player):
    def __init__(self, color):
        super().__init__(color)

        self.calculation_depth = 3

        self.h_for_filter = build_heuristic(self.color, scorer_type="count_with_move")

        self.max_workers = 3

        self.pool: Pool | None = None
        self.use_pool = True

        self.is_win_h = build_heuristic(color=0, scorer_type="bin")

    @cache
    def next_positions(self, board: Board, color: int) -> list[Board]:
        possible_moves_set = set()

        stones_coords = np.argwhere(board.position != self.empty_color)
        for my_stone_coords in stones_coords:
            for neighbour_coords in board.get_point_neighbours_to_all_directions(
                *my_stone_coords
            ):
                if board.is_point_empty(neighbour_coords[0], neighbour_coords[1]):
                    possible_moves_set.add(tuple(neighbour_coords))

        possible_moves_set.update(
            [
                tuple(p)
                for p in board.get_center_square_points()
                if board.is_point_empty(p[0], p[1])
            ]
        )
        return [
            board.get_board_after_move(m[0], m[1], color) for m in possible_moves_set
        ]

    def check_win(self, next_positions: list[Board]):
        win_check = [self.is_win_h(None, p) for p in next_positions]
        if self.color in win_check:
            return np.inf
        elif self.opponent_color in win_check:
            return -np.inf
        else:
            return None

    @cached(
        cache={},
        key=lambda se, position, move_color, depth, alpha, beta: (
            position,
            move_color,
            depth,
        ),
    )
    def minimax(
        self, position: Board, move_color: int, depth: int, alpha: float, beta: float
    ) -> tuple[float, Board | None]:
        if depth == 0:
            h = self.h(move_color, position)
            return h, None

        this_layer_best_next_position = None

        if self.color == move_color:
            if alpha == np.inf:
                return alpha, None
            this_layer_best_val = -np.inf

            next_positions = self.next_positions(position, self.color)

            if (
                position.move_idx > 9
                and (win_res := self.check_win(next_positions)) is not None
            ):
                return win_res, None

            if depth > 1:
                for next_position in sorted(
                    next_positions,
                    key=lambda x: -self.h_for_filter(self.opponent_color, x),
                ):
                    val, _ = self.minimax(
                        next_position, self.opponent_color, depth - 1, alpha, beta
                    )
                    if val == np.inf:
                        return val, next_position

                    if val > this_layer_best_val:
                        this_layer_best_val = val
                        this_layer_best_next_position = next_position

                    if this_layer_best_val > alpha:
                        alpha = this_layer_best_val

                    if beta <= alpha:
                        break
            else:
                for val in self.map_fn(
                    partial(self.h, self.opponent_color), next_positions
                ):
                    if val == np.inf:
                        return val, this_layer_best_next_position

                    if val > this_layer_best_val:
                        this_layer_best_val = val

                    if this_layer_best_val > alpha:
                        alpha = this_layer_best_val

                    if beta <= alpha:
                        break
        else:
            if beta == -np.inf:
                return beta, None
            this_layer_best_val = np.inf

            next_positions = self.next_positions(position, self.opponent_color)

            if (
                position.move_idx > 9
                and (win_res := self.check_win(next_positions)) is not None
            ):
                return win_res, None

            if depth > 1:
                for next_position in sorted(
                    next_positions, key=lambda x: self.h_for_filter(self.color, x)
                ):
                    val, _ = self.minimax(
                        next_position, self.color, depth - 1, alpha, beta
                    )
                    if val == -np.inf:
                        return val, next_position

                    if val < this_layer_best_val:
                        this_layer_best_val = val
                        this_layer_best_next_position = next_position

                    if this_layer_best_val < beta:
                        beta = this_layer_best_val

                    if beta <= alpha:
                        break
            else:
                for val in self.map_fn(partial(self.h, self.color), next_positions):
                    if val == -np.inf:
                        return val, this_layer_best_next_position

                    if val < this_layer_best_val:
                        this_layer_best_val = val

                    if this_layer_best_val < beta:
                        beta = this_layer_best_val

                    if beta <= alpha:
                        break

        return this_layer_best_val, this_layer_best_next_position

    # def minimax_maximizer(self, position: Board, move_color: int, depth: int, alpha: float, beta: float):

    def get_move(self, position: Board) -> tuple[int, int]:
        _, best_next_position = self.minimax(
            position, self.color, self.calculation_depth + 1, -np.inf, np.inf
        )  # чтобы закешировать minimax

        return first_non_equal_element_coords(position, best_next_position)

    def start_game(self):
        self.pool = Pool(max_workers=self.max_workers)
        if self.use_pool:
            self.map_fn = self.pool.map
        else:
            self.map_fn = map

    def end_game(self):
        self.pool.shutdown()
