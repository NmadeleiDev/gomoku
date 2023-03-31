import argparse
import logging
from typing import Literal

from gameplay.graphical import VisualGameplay
from gameplay.terminal import TerminalGameplay
from player.ai import AIPlayer
from player.ai_benchmark import BenchmarkPlayer
from player.base import Player
from player.human import HumanPlayer


def init_logging():
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s ~ %(funcName)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S:",
        level=logging.DEBUG,
    )


def create_players(
    p_types: list[Literal["human", "AI", "bm"]]
) -> tuple[Player, Player]:
    player_colors = [1, -1]

    p_types_dict = {"human": HumanPlayer, "AI": AIPlayer, "bm": BenchmarkPlayer}

    return tuple([p_types_dict[p_type](player_colors.pop()) for p_type in p_types])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player1", "-p1", type=str, dest="p1", default="human")
    parser.add_argument("--player2", "-p2", type=str, dest="p2", default="AI")
    parser.add_argument(
        "--gameplay",
        "-g",
        type=str,
        choices=["visual", "terminal"],
        dest="g",
        default="visual",
    )
    return parser.parse_args()


def start_game(
    player_1: Player,
    player_2: Player,
    gameplay_class: type[VisualGameplay] | type[TerminalGameplay],
):
    player_1.start_game()
    player_2.start_game()

    vis = gameplay_class(player_1, player_2)

    try:
        vis.start()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, clearing resources and exiting...")
        player_1.end_game()
        player_2.end_game()
    except Exception as e:
        logging.exception(e)
        player_1.end_game()
        player_2.end_game()

    player_1.end_game()
    player_2.end_game()


def main():
    init_logging()
    args = parse_args()

    gameplay_classes_dict = {"visual": VisualGameplay, "terminal": TerminalGameplay}

    start_game(*create_players([args.p1, args.p2]), gameplay_classes_dict[args.g])


if __name__ == "__main__":
    main()
