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
                raise ValueError(f"Coordinates must be in [0:{Board.size - 1}]")
        return x, y

    def start(self):
        self.call_game_iteration()

    def call_game_iteration(self):
        for result in self.game_iterator_instance:
            if result is None:
                continue
            break

    def game_iterator(self):
        clear_previous_game_logs()

        winner_color = None

        players_chars = {
            self.player_1.color: "X",
            self.player_2.color: "O",
        }
        players_hs = {p.color: p.h for p in self.players}
        players_timers = {p.color: [] for p in self.players}

        winner_heuristic = build_heuristic(0, Heuristics.bin)

        while winner_color is None:
            self.print_info_before_move(self.board, players_chars)
            joblib.dump(self.board, f"./logs/board_at_move_{self.move_idx}.joblib")

            try:
                time_start = datetime.now()
                move_x, move_y = self.active_player.get_move(self.board)
                players_timers[self.active_player.color].append(
                    datetime.now() - time_start
                )
                board_new = self.board.get_board_after_move(
                    move_x, move_y, self.active_player.color
                )
                if board_new.update_double_free_three_count_and_check_if_violated(
                    self.active_player.free_three_counter
                ):
                    print("Move violates double free three rule, try again")
                    continue
            except ValueError as e:
                print(f"Failed to get move: {e}, try again")
                continue

            self.board = board_new

            scores = {
                k: players_hs[k](self.active_player.color, self.board)
                for k in players_chars.keys()
            }

            self.print_info_after_move(
                self.board,
                players_chars,
                players_timers,
                (move_x, move_y),
                scores,
            )

            winner_color = self.board.winner(winner_heuristic)

            self.increment_move_index()
            yield None

        self.print_end_game_info(winner_color, players_chars)
        self.board.players_chars = players_chars
        print(self.board, '\n')
        yield winner_color

    def print_info_before_move(self, board, players_chars):
        print(
            f"Move #{self.move_idx // 2} / {players_chars[self.active_player.color]}. Current board is:"
        )
        board.players_chars = players_chars
        print(board, '\n')

    def print_info_after_move(
        self,
        board,
        players_chars,
        players_timers,
        move: tuple[int, int],
        scores: dict[int, int],
    ):
        move_x, move_y = move

        print(
            f'\nPlayer "{players_chars[self.active_player.color]}" is playing [{move_x}, {move_y}] '
            f"after {players_timers[self.active_player.color][-1]}\n"
            f"Mean time for move for player {players_chars[self.active_player.color]} "
            f"= {np.mean(players_timers[self.active_player.color])}"
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
