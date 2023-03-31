from datetime import datetime

import joblib
import numpy as np

from board import Board
from gameplay.base import BaseGameplay
from gameplay.utils import clear_previous_game_logs
from heuristics.sliding import Heuristics, build_heuristic


class TerminalGameplay(BaseGameplay):
    def __init__(self, player_1, player_2):
        super().__init__(player_1, player_2)

    def get_move(self, position) -> tuple[int, int]:
        move_line = input(
            f"Type X and Y coordinates of the move (must be from 0 to {Board.size - 1}) :: "
        )
        move = move_line.strip().split()
        try:
            x, y = [int(i.strip()) for i in move]
        except Exception:
            raise ValueError("Invalid input")

        for inp in (x, y):
            if inp >= Board.size or inp < 0:
                raise ValueError("Coordinates must be in [0:19]")
        return x, y

    def start(self):
        self.call_game_iteration(True)

    def call_game_iteration(self, do_wait_for_next_move):
        for result in self.game_iterator_instance:
            if result is None:
                continue
            break

    def game_iterator(self):
        clear_previous_game_logs()

        players = [self.player_1, self.player_2]

        board = Board()

        winner_color = None

        current_player_idx = 0
        players_chars = {
            self.player_1.color: "X",
            self.player_2.color: "O",
        }
        players_hs = {p.color: p.h for p in players}
        players_timers = {p.color: [] for p in players}

        winner_heuristic = build_heuristic(0, Heuristics.bin)

        move_idx = 0

        while winner_color is None:
            current_player = players[current_player_idx]
            self.print_info_before_move(board, current_player, players_chars)
            joblib.dump(board, f"./logs/board_at_move_{move_idx}.joblib")

            try:
                time_start = datetime.now()
                move_x, move_y = current_player.get_move(board)
                players_timers[current_player.color].append(datetime.now() - time_start)
                board_new = board.get_board_after_move(
                    move_x, move_y, current_player.color
                )
                if board_new.update_double_free_three_count_and_check_if_violated(
                    current_player.free_three_counter
                ):
                    print("Move violates double free three rule, try again")
                    continue
            except ValueError as e:
                print(f"Failed to get move: {e}, try again")
                continue

            board = board_new

            scores = {
                k: players_hs[k](current_player.color, board)
                for k in players_chars.keys()
            }

            self.print_info_after_move(
                board,
                current_player,
                players_chars,
                players_timers,
                (move_x, move_y),
                scores,
            )

            winner_color = board.winner(winner_heuristic)

            move_idx += 1

            current_player_idx = (current_player_idx + 1) % 2
            yield None

        self.print_end_game_info(winner_color, players_chars)
        board.print_board(players_chars)
        yield winner_color

    @staticmethod
    def print_info_before_move(board, current_player, players_chars):
        print(
            f'\nMove by player "{players_chars[current_player.color]}" , '
            f"{type(current_player)}). Current board is:"
        )
        board.print_board(players_chars)
        print()

    @staticmethod
    def print_info_after_move(
        board,
        current_player,
        players_chars,
        players_timers,
        move: tuple[int, int],
        scores: dict[int, int],
    ):
        move_x, move_y = move

        print(
            f'\nPlayer "{players_chars[current_player.color]}" is playing [{move_x}, {move_y}] '
            f"after {players_timers[current_player.color][-1]}\n"
            f"Mean time for move for player {players_chars[current_player.color]} "
            f"= {np.mean(players_timers[current_player.color])}"
        )

        print(
            f'Scores are: {", ".join([players_chars[k] + "=" + str(v) for k, v in scores.items()])}\n'
            "Captures are:",
            {players_chars[k]: v * 2 for k, v in board.captures.items()},
            "\n",
        )

    @staticmethod
    def print_end_game_info(winner_color, players_chars):
        print(f'Game finished, player "{players_chars[winner_color]}" won!')
