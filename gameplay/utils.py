import os


def clear_previous_game_logs():
    if not os.path.isdir("../logs"):
        os.mkdir("../logs")

    for name in os.listdir("../logs"):
        if name.endswith(".joblib"):
            os.unlink(os.path.join("../logs", name))
