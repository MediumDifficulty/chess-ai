import chess
import datetime

date = datetime.datetime.now()

pgn = f"""[Event "Test"]
[Site "Test Country"]
[Date "{date.strftime("%Y.%m.%d")}"]
[Round "1"]
[White "Player 1"]
[Black "Player 2"]
[Result "1-0"]
"""

with open("game.uci.txt", "r") as file:
    content = file.read()
    board = chess.Board()

    for (i, line) in enumerate(content.splitlines()):
        move = chess.Move.from_uci(line)
        pgn += f"{i+1}. {board.san(move)}\n"
        board.push(move)

with open("game.pgn.txt", "w") as file:
    file.write(pgn)
