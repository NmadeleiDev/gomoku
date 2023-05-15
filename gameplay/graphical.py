import logging
import tkinter
import tkinter as tk
from datetime import datetime, timedelta
from queue import Queue
from tkinter import Tk

import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageTk
from traig_client.client import MetricTypeEnum as TraigMetricTypeEnum
from traig_client.client import get_client as traig_client

from board import Board
from exceptions import IllegalMove
from gameplay.base import BaseGameplay
from gameplay.utils import clear_previous_game_logs
from player.human import HumanPlayer

SIZE = 650
WINDOW_XY = np.array([SIZE, SIZE])

BOARD_ZOOM_FACTOR = 0.85
BOARD_ZOOM = np.array(WINDOW_XY * BOARD_ZOOM_FACTOR, dtype=int)
ARMY_SIZE = np.array(BOARD_ZOOM * 0.04, dtype=int)
OFFSET = (SIZE - SIZE * BOARD_ZOOM_FACTOR) * 1.4 / 2
PLAYING_BOARD_PART_SIZE = SIZE * BOARD_ZOOM_FACTOR * 0.984
STEP_SIZE = PLAYING_BOARD_PART_SIZE / Board.size
FIELD_SIZE = STEP_SIZE / 2


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


class VisualGameplay(BaseGameplay):
    def __init__(self, player_1, player_2):
        super().__init__(player_1, player_2)
        self.player_timers = None
        self.player_stone_img_dict = None
        self.fields = None
        self.canvas = None
        self.total_time_player_2_label = None
        self.total_time_player_1_label = None
        self.mean_time_player_2_label = None
        self.mean_time_player_1_label = None
        self.captures_player_2_label = None
        self.captures_player_1_label = None
        self.player_2_frame = None
        self.player_1_frame = None
        self.players_frame = None
        self.move_idx_label = None
        self.frame = None

        self.root = Tk()
        self.root.title("Gomoku")
        self.root.geometry(f"{WINDOW_XY[0]}x{WINDOW_XY[1] + 100}")

        self.moves_queue = None

        self.img_stone_black_opaque = None
        self.img_stone_black = None
        self.img_stone_white_opaque = None
        self.img_stone_white = None
        self.img_board = None
        self.img_error = None
        self.player_timer_labels = None

        self.after_cb_id = None

        self.game_start_time = None

        self.draw_initial_state()

    def draw_initial_state(self):
        self.frame = tk.Frame(self.root, width=WINDOW_XY[0], height=100)
        self.move_idx_label = tk.Label(
            self.frame, text="Move #0", font=("Helvetica", 18, "bold")
        )
        self.move_idx_label.pack(side="top")
        controls_frame = tk.Frame(self.frame, width=WINDOW_XY[0] // 2, height=100)
        controls_frame.pack(side="left")
        tk.Button(controls_frame, text="AI help", command=self.ai_help).pack(side="top")
        tk.Button(controls_frame, text="Reset", command=self.reset_game).pack(
            side="top"
        )
        tk.Button(controls_frame, text="Exit", command=self.exit).pack(side="top")

        self.players_frame = tk.Frame(self.frame, width=WINDOW_XY[0], height=100)
        self.players_frame.pack(side="bottom")

        self.player_1_frame = tk.Frame(
            self.players_frame,
            width=WINDOW_XY[0] // 2,
            height=100,
            highlightthickness=4,
            highlightbackground="green",
            padx=20,
            pady=10,
        )
        self.player_2_frame = tk.Frame(
            self.players_frame,
            width=WINDOW_XY[0] // 2,
            height=100,
            highlightthickness=4,
            highlightbackground="gray",
            padx=20,
            pady=10,
        )
        self.player_1_frame.pack(side="left")
        self.player_2_frame.pack(side="right")
        self.frame.pack()

        tk.Label(
            self.player_1_frame,
            text=f"{self.player_color_str(self.player_1.color)} ({type(self.player_1).__name__.replace('Player', '')})",
            font=("Helvetica", 18, "bold"),
        ).pack(side="top")
        tk.Label(
            self.player_2_frame,
            text=f"{self.player_color_str(self.player_2.color)} ({type(self.player_2).__name__.replace('Player', '')})",
            font=("Helvetica", 18, "bold"),
        ).pack(side="top")
        self.captures_player_1_label = tk.Label(self.player_1_frame, text="Captures: 0")
        self.captures_player_1_label.pack(side="top")
        self.captures_player_2_label = tk.Label(self.player_2_frame, text="Captures: 0")
        self.captures_player_2_label.pack(side="top")

        self.mean_time_player_1_label = tk.Label(
            self.player_1_frame, text="Mean time: 0"
        )
        self.mean_time_player_1_label.pack(side="bottom")
        self.mean_time_player_2_label = tk.Label(
            self.player_2_frame, text="Mean time: 0"
        )
        self.mean_time_player_2_label.pack(side="bottom")

        self.total_time_player_1_label = tk.Label(
            self.player_1_frame, text="Total time: 0"
        )
        self.total_time_player_1_label.pack(side="bottom")
        self.total_time_player_2_label = tk.Label(
            self.player_2_frame, text="Total time: 0"
        )
        self.total_time_player_2_label.pack(side="bottom")

        self.canvas = tk.Canvas(self.root, width=WINDOW_XY[0], height=WINDOW_XY[1])
        self.canvas.configure(background="#8baabf")
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.move_callback)

        self.load_tk_images()
        self.player_stone_img_dict = {
            self.player_1.color: [self.img_stone_white, self.img_stone_white_opaque],
            self.player_2.color: [self.img_stone_black, self.img_stone_black_opaque],
        }

        self.canvas.create_image(
            WINDOW_XY[0] // 2, WINDOW_XY[1] // 2, anchor="center", image=self.img_board
        )

        self.player_timer_labels = {
            self.player_1.color: [
                self.total_time_player_1_label,
                self.mean_time_player_1_label,
            ],
            self.player_2.color: [
                self.total_time_player_2_label,
                self.mean_time_player_2_label,
            ],
        }

    def player_color_str(self, color):
        return "White" if color == self.player_1.color else "Black"

    def pre_game_init(self):
        super().pre_game_init()
        self.player_timers = {p.color: 0 for p in [self.player_1, self.player_2]}
        self.fields = [[None for _ in range(Board.size)] for _ in range(Board.size)]
        self.moves_queue = Queue(1)

    def start(self):
        self.after_cb_id = self.root.after(100, self.call_game_iteration)
        self.game_start_time = datetime.now()
        self.root.mainloop()

    def close_window(self):
        self.root.destroy()

    def ai_help(self):
        self.moves_queue.put(self.get_move_from_ai())

    def reset_game(self):
        self.root.after_cancel(self.after_cb_id)
        self.frame.destroy()
        self.canvas.destroy()
        self.pre_game_init()
        self.game_iterator_instance = self.game_iterator()
        self.draw_initial_state()
        self.after_cb_id = self.root.after(100, self.call_game_iteration)

    def exit(self):
        self.root.destroy()
        self.root.quit()

    def call_game_iteration(self):
        self.move_idx_label.configure(
            text=f"Move #{self.move_idx // 2} / {self.player_color_str(self.active_player.color)}"
        )
        if isinstance(self.active_player, HumanPlayer) and self.moves_queue.empty():
            self.after_cb_id = self.root.after(100, self.call_game_iteration)
        else:
            winner_color = next(self.game_iterator_instance)
            if winner_color is not None:
                winner_color_name = self.player_color_str(winner_color)
                tk.Label(
                    self.root,
                    text=f"{winner_color_name} won!",
                    font=("Helvetica", 24, "bold"),
                    padx=10,
                    pady=10,
                ).place(anchor=tkinter.CENTER, relx=0.5, rely=0.5)
                return
            self.after_cb_id = self.root.after(10, self.call_game_iteration)

    def move_callback(self, event):
        if not self.moves_queue.empty():
            print("Queue not empty")
            return
        i = int((event.x - OFFSET + FIELD_SIZE) // STEP_SIZE)
        j = int((event.y - OFFSET + FIELD_SIZE) // STEP_SIZE)

        if i < 0 or j < 0 or i >= len(self.fields) or j >= len(self.fields[i]):
            print("Wrong move")
            return

        self.moves_queue.put((i, j))

    def place_stone(self, i: int, j: int, color: int):
        if i >= len(self.fields) or j >= len(self.fields[i]):
            print("Wrong move")
            return

        stone_img_ref = self.canvas.create_image(
            OFFSET + i * STEP_SIZE,
            OFFSET + j * STEP_SIZE,
            image=self.player_stone_img_dict[color][0],
        )

        self.fields[i][j] = stone_img_ref

    def place_error_img(self, i: int, j: int):
        if i >= len(self.fields) or j >= len(self.fields[i]):
            print("Wrong move")
            return

        err_img_ref = self.canvas.create_image(
            OFFSET + i * STEP_SIZE,
            OFFSET + j * STEP_SIZE,
            image=self.img_error,
        )

        self.root.after(3000, self.remove_img_from_canvas, err_img_ref)

    def remove_img_from_canvas(self, ref: int):
        self.canvas.delete(ref)

    def draw_stones(self, board: Board):
        for i in range(board.size):
            for j in range(board.size):
                if self.fields[i][j] is not None:
                    self.canvas.delete(self.fields[i][j])
                if board.position[i][j] != Board.empty_color:
                    self.place_stone(i, j, board.position[i][j])

    def game_iterator(self):
        clear_previous_game_logs()

        winner_color = None

        players_timers = {p.color: [] for p in self.players}

        while winner_color is None:
            self.board.dump(f'at_move_{self.move_idx}')

            try:
                if isinstance(self.active_player, HumanPlayer):
                    total_time_passed = datetime.now() - self.game_start_time
                    time_sum = (
                        total_time_passed
                        - sum(players_timers[self.passive_player.color], timedelta())
                    ).total_seconds()
                    prev_time_sum = sum(
                        players_timers[self.active_player.color], timedelta()
                    ).total_seconds()
                    last_move_time = time_sum - prev_time_sum
                    move_x, move_y = self.active_player.get_move(self.board)
                    players_timers[self.active_player.color].append(
                        timedelta(seconds=last_move_time)
                    )
                    self.player_timer_labels[self.active_player.color][0].configure(
                        text=f"Total time: {time_sum:.2f}"
                    )
                    self.player_timer_labels[self.active_player.color][1].configure(
                        text=f"Mean time: {(time_sum / len(players_timers[self.active_player.color])):.2f}"
                    )
                else:
                    time_start = datetime.now()
                    move_x, move_y = self.active_player.get_move(self.board)
                    players_timers[self.active_player.color].append(
                        datetime.now() - time_start
                    )
                    time_sum = sum(
                        players_timers[self.active_player.color], timedelta()
                    ).total_seconds()
                    self.player_timer_labels[self.active_player.color][0].configure(
                        text=f"Total time: {time_sum:.2f}"
                    )
                    self.player_timer_labels[self.active_player.color][1].configure(
                        text=f"Mean time: {(time_sum / len(players_timers[self.active_player.color])):.2f}"
                    )
                board_new = self.board.get_board_after_move(
                    move_x, move_y, self.active_player.color
                )
                if board_new.update_double_free_three_count_and_check_if_violated(
                    self.active_player.free_three_counter
                ):
                    print("Move violates double free three rule, try again")
                    self.place_error_img(move_x, move_y)
                    yield None
                    continue
            except IllegalMove as e:
                print(f"Failed to get move: {e}, try again")
                yield None
                continue

            self.board = board_new

            for color, n_captures in self.board.captures.items():
                if color == self.player_1.color:
                    self.captures_player_1_label.configure(
                        text=f"Captures: {n_captures * 2}"
                    )
                elif color == self.player_2.color:
                    self.captures_player_2_label.configure(
                        text=f"Captures: {n_captures * 2}"
                    )

            self.draw_stones(self.board)

            winner_color = self.board.winner(self.winner_heuristic)

            self.increment_move_index()
            if not winner_color:
                yield None

        yield winner_color

    def load_tk_images(self):
        error = Image.open("./img/error.png").resize(ARMY_SIZE, Image.ANTIALIAS)
        self.img_error = ImageTk.PhotoImage(error)
        board = Image.open("./img/board.png").resize(BOARD_ZOOM, Image.ANTIALIAS)
        self.img_board = ImageTk.PhotoImage(board)
        horde = Image.open("./img/white.png").resize(ARMY_SIZE, Image.ANTIALIAS)
        self.img_stone_white = ImageTk.PhotoImage(horde)
        horde.putalpha(ImageEnhance.Brightness(horde.split()[3]).enhance(0.5))
        self.img_stone_white_opaque = ImageTk.PhotoImage(horde)
        alliance = Image.open("./img/black.png").resize(ARMY_SIZE, Image.ANTIALIAS)
        self.img_stone_black = ImageTk.PhotoImage(alliance)
        alliance.putalpha(ImageEnhance.Brightness(alliance.split()[3]).enhance(0.5))
        self.img_stone_black_opaque = ImageTk.PhotoImage(alliance)

    def get_move(self, *args) -> tuple[int, int]:
        return self.moves_queue.get()
