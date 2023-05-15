import os
from functools import cache, partial
from multiprocessing import Manager
from multiprocessing.managers import ValueProxy
from multiprocessing.pool import AsyncResult

import numpy as np
from cachetools import cached

from board import Board
from heuristics.sliding import Heuristics, build_heuristic
from player.base import Player

free_three_counter = build_heuristic(None, Heuristics.free_three, line_len_to_analyze=6)


def yield_completed(positions: list, async_results: list[AsyncResult]):
    async_results_dict = {i: v for i, v in enumerate(async_results)}
    while len(async_results_dict) > 0:
        to_pop = None
        for i, as_res in async_results_dict.items():
            if as_res.ready():
                to_pop = i
                yield as_res.get(), positions[i]
        if to_pop is not None:
            async_results_dict.pop(to_pop)


@cache
def get_next_positions(board: Board, color: int, h=None) -> list[Board]:
    possible_moves_set = set()

    for stone_coords in np.argwhere(board.position != board.empty_color):
        for neighbour_coords in board.get_point_neighbours_coords_to_all_directions(
            *stone_coords, at_distance=1
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
        result = [
            board.get_board_after_move(m[0], m[1], color) for m in possible_moves_set
        ]
    else:
        result = [
            board.get_board_after_move(m[0], m[1], color).compute_h_for_self(h)
            for m in possible_moves_set
        ]

    return [
        x
        for x in result
        if not x.update_double_free_three_count_and_check_if_violated(
            free_three_counter
        )
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
    alpha: float | ValueProxy,
    beta: float | ValueProxy,
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

    if next_positions[0].h_val == win_value:
        next_positions = [next_positions[0]]

    this_layer_best_score = -win_value
    this_layer_best_next_move = None

    if pool is None and isinstance(alpha, ValueProxy):
        alpha = alpha.value
        beta = beta.value

    next_minimax_partial = partial(
        minimax, 
        not is_maximizer,
        depth - 1,
        alpha,
        beta,
        maximizer_color,
        minimizer_color,
        h_func)

    if is_maximizer:
        if pool is not None:
            results = [
                pool.apply_async(
                    next_minimax_partial,
                    args=(next_position,),
                )
                for next_position in next_positions
            ]

            for (score, _), next_position in yield_completed(next_positions, results):
                if score > this_layer_best_score or this_layer_best_next_move is None:
                    this_layer_best_score = score
                    this_layer_best_next_move = next_position.from_move

                if this_layer_best_score > alpha.value:
                    alpha.value = this_layer_best_score

                if beta.value <= alpha.value:
                    break

        else:
            for next_position in next_positions:
                score, _ = next_minimax_partial(next_position)

                if score > this_layer_best_score or this_layer_best_next_move is None:
                    this_layer_best_score = score
                    this_layer_best_next_move = next_position.from_move

                if this_layer_best_score > alpha:
                    alpha = this_layer_best_score

                if beta <= alpha:
                    break
    else:
        for next_position in next_positions:
            score, _ = next_minimax_partial(next_position)

            if score < this_layer_best_score or this_layer_best_next_move is None:
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

        self.calculation_depth = int(os.getenv("DEPTH", "3"))

        self.h = build_heuristic(self.color, Heuristics.count)

        self.max_workers = os.cpu_count()

    def get_move(self, position: Board) -> tuple[int, int]:
        _, best_next_move = minimax(
            True,
            self.calculation_depth,
            self.manager.Value("i", -np.inf),
            self.manager.Value("i", np.inf),
            self.color,
            self.opponent_color,
            self.h,
            position,
            pool=self.pool,
        )

        return best_next_move

    def start_game(self):
        self.manager = Manager()
        self.pool = self.manager.Pool(processes=self.max_workers)

    def end_game(self):
        self.manager.shutdown()
