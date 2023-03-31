from abc import ABC


class BaseGameplay(ABC):
    def __init__(self, player_1, player_2):
        self.player_1 = player_1
        self.player_2 = player_2

        self.player_1.set_move_getter(self.get_move)
        self.player_2.set_move_getter(self.get_move)

        self.game_iterator_instance = self.game_iterator()

    def get_move(self, position) -> tuple[int, int]:
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def move_callback(self, event):
        raise NotImplementedError()

    def call_game_iteration(self, do_wait_for_next_move):
        raise NotImplementedError()

    def game_iterator(self):
        raise NotImplementedError()
