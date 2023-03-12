import logging
import os
from datetime import datetime

import joblib
import numpy as np
from traig_client.client import MetricTypeEnum as TraigMetricTypeEnum
from traig_client.client import get_client as traig_client

from board import Board
from heuristics.sliding import build_heuristic
from player.base import Player


def clear_previous_game_logs():
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")

    for name in os.listdir("./logs"):
        if name.endswith(".joblib"):
            os.unlink(os.path.join("./logs", name))


def init_traig_client():
    traig_client().init_metrics(
        player_X_mean_move_time=TraigMetricTypeEnum.mean.value,
        player_O_mean_move_time=TraigMetricTypeEnum.mean.value,
        mean_move_time=TraigMetricTypeEnum.mean.value,
        n_moves=TraigMetricTypeEnum.count.value,
        who_won=TraigMetricTypeEnum.value.value,
        **{
            "color_-1_calc_depth": TraigMetricTypeEnum.value.value,
            "color_1_calc_depth": TraigMetricTypeEnum.value.value,
        },
    )

    logging.debug(f"traig client is {traig_client()}")


def play_game(player_1: Player, player_2: Player):
    init_traig_client()

    clear_previous_game_logs()

    board = Board()

    winner_color = None

    players = [player_1, player_2]
    current_player_idx = 0
    players_chars = {
        player_1.color: "X",
        player_2.color: "O",
    }
    players_hs = {p.color: p.h for p in players}
    players_timers = {p.color: [] for p in players}

    winner_heuristic = build_heuristic(color=0, scorer_type="bin")

    move_idx = 0

    while winner_color is None:
        current_player = players[current_player_idx]
        print(
            f'\n#{move_idx} :: Move by player "{players_chars[current_player.color]}". Current board is:'
        )
        board.print_board(players_chars)
        print()
        joblib.dump(board, f"./logs/board_at_move_{move_idx}.joblib")

        try:
            time_start = datetime.now()
            move_x, move_y = current_player.get_move(board)
            players_timers[current_player.color].append(datetime.now() - time_start)
            board = board.get_board_after_move(move_x, move_y, current_player.color)
        except ValueError as e:
            print(f"Failed to get move: {e}")
            print("Try again")
            continue

        print(
            f'\nPlayer "{players_chars[current_player.color]}" is playing [{move_x}, {move_y}] '
            f"after {players_timers[current_player.color][-1]}"
        )
        print(
            f"Mean time for move for player {players_chars[current_player.color]} "
            f"= {np.mean(players_timers[current_player.color])}"
        )

        traig_client().update_metrics(
            **{
                f"player_{players_chars[current_player.color]}_mean_move_time": (
                    players_timers[current_player.color][-1]
                ).total_seconds(),
                "mean_move_time": players_timers[current_player.color][
                    -1
                ].total_seconds(),
                "n_moves": 1,
            }
        )

        scores = {
            k: players_hs[k](current_player.color, board) for k in players_chars.keys()
        }
        print(
            f'Scores are: {", ".join([players_chars[k] + "=" + str(v) for k, v in scores.items()])}\n'
        )

        if (winner := winner_heuristic(current_player.color, board)) != 0:
            winner_color = winner

        move_idx += 1

        # TODO: считать captures, по ним определять победителя
        current_player_idx = (current_player_idx + 1) % 2

    print(f'Game finished, player "{players_chars[winner_color]}" won!')
    traig_client().update_metrics(who_won=players_chars[winner_color])
    board.print_board(players_chars)


def start_game(player_1: Player, player_2: Player):
    player_1.start_game()
    player_2.start_game()

    try:
        play_game(player_1, player_2)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, clearing resources and exiting...")
        player_1.end_game()
        player_2.end_game()

    player_1.end_game()
    player_2.end_game()
