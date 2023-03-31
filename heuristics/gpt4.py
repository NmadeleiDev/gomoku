def count_open_sequences(board, player):
    counts = [0] * 6
    for x in range(board.size):
        for y in range(board.size):
            if board.board[x][y] == player:
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    length = 0
                    nx, ny = x + dx, y + dy
                    while (
                        0 <= nx < board.size
                        and 0 <= ny < board.size
                        and board.board[nx][ny] == player
                    ):
                        length += 1
                        nx += dx
                        ny += dy
                    if (
                        0 <= nx - dx - dx < board.size
                        and 0 <= ny - dy - dy < board.size
                        and board.board[nx - dx - dx][ny - dy - dy] == 0
                    ):
                        counts[length] += 1
    return counts


def heuristic(board, player):
    counts = count_open_sequences(board, player)
    opponent_counts = count_open_sequences(board, 3 - player)

    weights = [0, 1, 10, 100, 1000, 10000]
    player_score = sum(counts[i] * weights[i] for i in range(len(counts)))
    opponent_score = sum(
        opponent_counts[i] * weights[i] for i in range(len(opponent_counts))
    )

    return player_score - opponent_score


def count_threats(board, player):
    counts = [0] * 5
    for x in range(board.size):
        for y in range(board.size):
            if board.board[x][y] == player:
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    length = 0
                    nx, ny = x + dx, y + dy
                    while (
                        0 <= nx < board.size
                        and 0 <= ny < board.size
                        and board.board[nx][ny] == player
                    ):
                        length += 1
                        nx += dx
                        ny += dy
                    if (
                        0 <= nx < board.size
                        and 0 <= ny < board.size
                        and board.board[nx][ny] == 0
                    ):
                        counts[length - 1] += 1
    return counts


def heuristic(board, player):
    counts = count_threats(board, player)
    opponent_counts = count_threats(board, 3 - player)

    weights = [1, 10, 100, 1000, 10000]
    player_score = sum(counts[i] * weights[i] for i in range(len(counts)))
    opponent_score = sum(
        opponent_counts[i] * weights[i] for i in range(len(opponent_counts))
    )

    return player_score - opponent_score
