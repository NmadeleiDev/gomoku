from board import Board
from player.base import Player


def terminal_input_move_getter(*args):
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


class HumanPlayer(Player):
    def __init__(self, color: int):
        super().__init__(color)

        self.set_move_getter(terminal_input_move_getter)

    def get_move(self, position: Board) -> tuple[int, int]:
        return self.move_getter(position)
