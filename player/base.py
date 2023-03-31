from abc import ABC

from board import Board
from heuristics.sliding import Heuristics, build_heuristic


class Player(ABC):
    empty_color = 0

    def __init__(self, color: int):
        self.color = color
        self.opponent_color = -color

        self.h = build_heuristic(self.color, Heuristics.count)
        self.free_three_counter = build_heuristic(
            self.color, Heuristics.free_three, line_len_to_analyze=6
        )

        self.move_getter = None

    def set_move_getter(self, move_getter):
        self.move_getter = move_getter

    def get_move(self, position: Board) -> tuple[int, int]:
        raise NotImplementedError()

    def start_game(self):
        pass

    def end_game(self):
        pass
