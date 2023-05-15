import os
from numba import jit, generated_jit

import numpy as np
from cachetools import cached
from traig_client.client import get_client as traig_client

from heuristics.sliding_jit import apply_scalar_heuristic, is_free_three, sum_for_homogeneous_line
from player.base import Player
from board_funcs import empty_color, get_board_after_move, get_move_between_boards, get_point_neighbours_coords_to_all_directions, get_position_hash, is_point_empty, get_center_square_points


@cached(
    cache={},
    key=lambda board, board_hash, color: (
        board_hash,
        color,
    ),
)
def get_free_three_count(board: np.ndarray, board_hash: int, color: int) -> int:
    return apply_scalar_heuristic(is_free_three, color, board, get_position_hash(board), line_len_to_analyze=6)


@cached(
    cache={},
    key=lambda board, board_hash, color: (
        board_hash,
        color,
    ),
)
def get_h_value(board: np.ndarray, board_hash: int, color: int) -> int:
    return apply_scalar_heuristic(sum_for_homogeneous_line, color, board, get_position_hash(board), line_len_to_analyze=5)


@cached(
    cache={},
    key=lambda board, color, board_hash: (
        board_hash,
        color,
    ),
)
@jit(nopython=True)
def get_next_positions(board: np.ndarray, color: int, board_hash: int) -> list[np.ndarray]:
    possible_moves_set = set()

    move_idx = 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if is_point_empty(board, i, j):
                continue
            move_idx += 1
            for neighbour_coords in get_point_neighbours_coords_to_all_directions(
                board, i, j, at_distance=1
            ):
                if is_point_empty(board, *neighbour_coords):
                    possible_moves_set.add(tuple(neighbour_coords))

    possible_moves_set.update(
        [
            tuple(p)
            for p in get_center_square_points(board)
            if is_point_empty(board, p[0], p[1])
        ]
    )
    result = [
        get_board_after_move(board, m[0], m[1], color) for m in possible_moves_set
        # TODO: добавить captures в get_board_after_move
    ]

    return result

    if move_idx > 8:
        prev_free_three_count = get_free_three_count(board, board_hash, color)

        return [
            x
            for x in result
            if (get_free_three_count(x, get_position_hash(x), -color) - prev_free_three_count) <= 1
        ]
    else:
        return result


@cached(
    cache={},
    key=lambda is_maximizer, depth, alpha, beta, maximizer_color, minimizer_color, position, position_hash: (
        position_hash,
        depth,
        is_maximizer,
    ),
)
@generated_jit(nopython=True)
def minimax(
    is_maximizer: bool,
    depth: int,
    alpha: float,
    beta: float,
    maximizer_color: int,
    minimizer_color: int,
    position: np.ndarray,
    position_hash: int,
) -> tuple[float, tuple[int, int] | None]:
    # if position.h_val is not None and position.h_val in (np.inf, -np.inf):
    #     return position.h_val, position.from_move

    if is_maximizer:
        move_color = maximizer_color
        win_value = np.inf
        next_move_color = minimizer_color
    else:
        move_color = minimizer_color
        win_value = -np.inf
        next_move_color = maximizer_color

    if depth == 0:
        return get_h_value(position, position_hash, next_move_color), None

    next_positions = sorted(
        get_next_positions(position, move_color, position_hash),
        key=lambda p: get_h_value(p, get_position_hash(p), next_move_color),
        reverse=is_maximizer,
    )

    if get_h_value(next_positions[0], get_position_hash(next_positions[0]), next_move_color) == win_value:
        next_positions = [next_positions[0]]

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
                next_position, get_position_hash(next_position))

            if score > this_layer_best_score or this_layer_best_next_move is None:
                this_layer_best_score = score
                this_layer_best_next_move = get_move_between_boards(position, next_position)

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
                next_position, get_position_hash(next_position))

            if score < this_layer_best_score or this_layer_best_next_move is None:
                this_layer_best_score = score
                this_layer_best_next_move = get_move_between_boards(position, next_position)

            if this_layer_best_score < beta:
                beta = this_layer_best_score

            if beta <= alpha:
                break

    return this_layer_best_score, this_layer_best_next_move


class AIPlayer(Player):
    def __init__(self, color):
        super().__init__(color)

        self.calculation_depth = int(os.getenv("DEPTH", "3"))

        self.max_workers = os.cpu_count()

        traig_client().update_metrics(
            **{f"color_{self.color}_calc_depth": self.calculation_depth}
        )

    def get_move(self, position) -> tuple[int, int]:
        position: np.ndarray = position.position
        position.flags.writeable = True

        _, best_next_move = minimax(
            True,
            self.calculation_depth,
            -np.inf,
            np.inf,
            self.color,
            self.opponent_color,
            position,
            get_position_hash(position),
        )

        return best_next_move

    def start_game(self):
        pass

    def end_game(self):
        pass
