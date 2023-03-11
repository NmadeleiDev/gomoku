from abc import ABC

from board import Board
from heuristics.sliding import build_heuristic


class Player(ABC):
    empty_color = 0

    def __init__(self, color: int):
        self.color = color
        self.opponent_color = -color

        self.h = build_heuristic(self.color)

    def get_move(self, position: Board) -> tuple[int, int]:
        raise NotImplementedError()

    def start_game(self):
        pass

    def end_game(self):
        pass
