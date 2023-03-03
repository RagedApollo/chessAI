import chess
import chess.engine
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci("stockfish.exe")
num_games = 100

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = MLPClassifier(max_iter=1000, learning_rate_init=0.01)

move_history = []
evaluation_history = []
for i in range(num_games):
    while not board.is_game_over():
        result = engine.play(board, chess.engine.Limit(time=0.00001))
        if not board.is_legal(result.move):
            continue
        board.push(result.move)
        print(board)
        move_history.append(result.move)
        encoded_moves = np.zeros((len(move_history), 64))
        for idx, move in enumerate(move_history):
            encoded_moves[idx, move.from_square] = 1
            encoded_moves[idx, move.to_square] = 1
        if board.result() == "1-0":
            evaluation = 1
            evaluation_history.append(evaluation)
            X = encoded_moves
            y = np.array(evaluation_history)
            model.fit(X, y)
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)
                print("File saved")
        elif board.result() == "0-1":
            evaluation = -1
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)
                print("File saved")
                evaluation_history.append(evaluation)
                X = encoded_moves
                y = np.array(evaluation_history)
                model.fit(X, y)
        else:
            evaluation = 0
            evaluation_history.append(evaluation)
            X = encoded_moves
            y = np.array(evaluation_history)
            model.fit(X, y)
        
    board.reset()
    move_history = []
    evaluation_history = []

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("File saved")

engine.quit()
