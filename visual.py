from board import Board


class Visual:
    def __init__(self):
        self.root = None

        self.update_loop()

    def update_stats(self, board: Board, new_stats: dict):
        """
        Принимает текущее состояние игры и отображает на доске
        :param board:
        :param new_stats:
        :return:
        """
        self.ui.update_info()

    def update_loop(self):
        self.root.after(1, self.update_stats)

    def get_move(self):
        pass
