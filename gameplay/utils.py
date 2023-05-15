import os


def clear_previous_game_logs():
    if not os.path.isdir("../logs"):
        os.mkdir("../logs")

    for name in os.listdir("../logs"):
        if name.endswith(".joblib"):
            os.remove(os.path.join("../logs", name))
            print("removed", os.path.join("../logs", name))
