import os

# from concurrent.futures import ProcessPoolExecutor as Pool
from functools import cache, partial

import numpy as np
from cachetools import cached
from traig_client.client import get_client as traig_client

from board import Board
from heuristics.sliding import build_heuristic
from player.base import Player

whos_win_h = build_heuristic(None, "bin")


class ValueStub:
    def __init__(self, v):
        self.value = v


@cache
def get_next_positions(board: Board, color: int, h=None) -> list[Board]:
    possible_moves_set = set()

    for stone_coords in np.argwhere(board.position != board.empty_color):
        for neighbour_coords in board.get_point_neighbours_to_all_directions(
            *stone_coords
        ):
            if board.is_point_empty(*neighbour_coords):
                possible_moves_set.add(tuple(neighbour_coords))

    possible_moves_set.update(
        [
            tuple(p)
            for p in board.get_center_square_points()
            if board.is_point_empty(p[0], p[1])
        ]
    )
    if h is None:
        return [
            board.get_board_after_move(m[0], m[1], color) for m in possible_moves_set
        ]
    else:
        return [
            board.get_board_after_move(m[0], m[1], color).compute_h_for_self(h)
            for m in possible_moves_set
        ]


@cached(
    cache={},
    key=lambda is_maximizer, depth, alpha, beta, maximizer_color, minimizer_color, h_func, position, pool=None: (
        position,
        depth,
        is_maximizer,
    ),
)
def minimax(
    is_maximizer: bool,
    depth: int,
    alpha: float,
    beta: float,
    maximizer_color: int,
    minimizer_color: int,
    h_func,
    position: Board,
    pool=None,
) -> tuple[float, tuple[int, int] | None]:
    if position.h_val is not None and position.h_val in (np.inf, -np.inf):
        return position.h_val, position.from_move

    if is_maximizer:
        move_color = maximizer_color
        win_value = np.inf
        next_move_color = minimizer_color
    else:
        move_color = minimizer_color
        win_value = -np.inf
        next_move_color = maximizer_color

    if depth == 0:
        return h_func(next_move_color, position), None

    next_positions = sorted(
        get_next_positions(position, move_color, h=partial(h_func, next_move_color)),
        key=lambda p: p.h_val,
        reverse=is_maximizer,
    )

    this_layer_best_score = -win_value
    this_layer_best_next_move = None

    if is_maximizer:
        for next_position in next_positions:
            score, _ = minimax(
                not is_maximizer,
                depth - 1,
                alpha,
                beta,
                maximizer_color,
                minimizer_color,
                h_func,
                next_position,
            )

            if score > this_layer_best_score:
                this_layer_best_score = score
                this_layer_best_next_move = next_position.from_move

            if this_layer_best_score > alpha:
                alpha = this_layer_best_score

            if beta <= alpha:
                break
    else:
        for next_position in next_positions:
            score, _ = minimax(
                not is_maximizer,
                depth - 1,
                alpha,
                beta,
                maximizer_color,
                minimizer_color,
                h_func,
                next_position,
            )

            if score < this_layer_best_score:
                this_layer_best_score = score
                this_layer_best_next_move = next_position.from_move

            if this_layer_best_score < beta:
                beta = this_layer_best_score

            if beta <= alpha:
                break

    return this_layer_best_score, this_layer_best_next_move


class AIPlayer(Player):
    def __init__(self, color):
        super().__init__(color)

        self.calculation_depth = 3

        # self.h_for_filter = build_heuristic(self.color, scorer_type="count_with_move")

        self.max_workers = os.cpu_count()

        traig_client().update_metrics(
            **{f"color_{self.color}_calc_depth": self.calculation_depth}
        )

    def get_move(self, position: Board) -> tuple[int, int]:
        _, best_next_move = minimax(
            True,
            self.calculation_depth,
            -np.inf,
            np.inf,
            self.color,
            self.opponent_color,
            self.h,
            position,
            # pool=self.pool
            pool=None,
        )

        return best_next_move

    def start_game(self):
        # self.manager = Manager()
        # self.pool = self.manager.Pool(processes=self.max_workers)
        # self.pool = None
        pass

    def end_game(self):
        # self.pool.shutdown()
        # self.manager.shutdown()
        # self.pool.join()
        # self.pool.close()
        pass
