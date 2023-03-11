import argparse
import logging
from typing import Literal

from gameplay import start_game
from player.ai import AIPlayer
from player.base import Player
from player.human import HumanPlayer


def init_logging():
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s ~ %(funcName)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S:",
        level=logging.DEBUG,
    )


def create_players(p_types: list[Literal["human", "AI"]]) -> tuple[Player, Player]:
    player_colors = [1, -1]

    p_types_dict = {
        "human": HumanPlayer,
        "AI": AIPlayer,
    }

    return tuple([p_types_dict[p_type](player_colors.pop()) for p_type in p_types])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player1", "-p1", type=str, dest="p1", default="human")
    parser.add_argument("--player2", "-p2", type=str, dest="p2", default="AI")
    return parser.parse_args()


def main():
    init_logging()
    args = parse_args()

    start_game(*create_players([args.p1, args.p2]))


if __name__ == "__main__":
    main()
