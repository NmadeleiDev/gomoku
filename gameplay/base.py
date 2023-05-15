from abc import ABC

from board import Board
from heuristics.sliding import Heuristics, build_heuristic
from player.ai import AIPlayer


class BaseGameplay(ABC):
    def __init__(self, player_1, player_2):
        self.board = None
        self.game_iterator_instance = None
        self.passive_player = None
        self.current_player_idx = None
        self.active_player = None
        self.move_idx = None
        self.player_1 = player_1
        self.player_2 = player_2

        self.players = [self.player_1, self.player_2]

        self.player_1.set_move_getter(self.get_move)
        self.player_2.set_move_getter(self.get_move)

        self.winner_heuristic = build_heuristic(0, Heuristics.bin)

    def pre_game_init(self):
        self.player_1.start_game()
        self.player_2.start_game()

        self.move_idx = 0
        self.current_player_idx = 0
        self.active_player = self.player_1
        self.passive_player = self.player_2

        self.game_iterator_instance = self.game_iterator()
        self.board = Board()

        print(f"player 1: type={type(self.player_1)}, h={type(self.player_1.h)}")
        print(f"player 2: type={type(self.player_2)}, h={type(self.player_2.h)}")

    def end_game(self):
        self.player_1.end_game()
        self.player_2.end_game()

    def increment_move_index(self):
        self.move_idx += 1
        self.passive_player = self.players[self.current_player_idx]
        self.current_player_idx = (self.current_player_idx + 1) % 2
        self.active_player = self.players[self.current_player_idx]

    def get_move_from_ai(self) -> tuple[int, int]:
        ai_player = AIPlayer(self.active_player.color)
        ai_player.start_game()
        move = ai_player.get_move(self.board)
        ai_player.end_game()
        return move

    def get_move(self, position) -> tuple[int, int]:
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def move_callback(self, event):
        raise NotImplementedError()

    def call_game_iteration(self):
        raise NotImplementedError()

    def game_iterator(self):
        """
        Initialization of this iteratior means start of the game.
        Yielding next iteration yields next move.
        Return value other than None means end of the game.
        """
        raise NotImplementedError()
