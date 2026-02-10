import os
import math
import chess

import numpy as np
import pandas as pd

# disables warnings from tensorfeed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split

data_paths = ["data/train-00000-of-00017.parquet"]
piece_map = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
             "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11}

def board_to_tensor(board: chess.Board):
    """
    Converts a chess.Board object into a (8, 8, 19) tensor input
    """
    tensor = np.zeros((8, 8, 19), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        tensor[rank, file, piece_map[piece.symbol()]] = 1.0
    
    if board.turn == chess.WHITE: tensor[:, :, 12] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE): tensor[:, :, 13] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[:, :, 14] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[:, :, 15] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[:, :, 16] = 1.0
    
    if board.ep_square:
        r, f = divmod(board.ep_square, 8)
        tensor[r, f, 17] = 1.0
    tensor[:, :, 18] = board.halfmove_clock / 100.0
    
    return tensor

def normalise_evaluation(cp, mate):
    """
    Normalises chess evaluations to a [-1, 1] range.
    """
    if not np.isnan(mate):
        # If there's a mate, it's either 1.0 (white wins) or -1.0 (black wins)
        return 1.0 if mate > 0 else -1.0
    
    # Scale centipawns. Using a tanh-style curve: 
    return math.tanh(cp / 5000)

def build_model():
    # Input shape is (8, 8, 19)
    inputs = layers.Input(shape=(8, 8, 19))

    # Convolutional layers to find spatial patterns (pinning, forks, pawn chains)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x) 
    
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)

    output = layers.Dense(1, activation="tanh")(x)

    optimiser = Adam(learning_rate=0.0005) 
    model = models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=optimiser, loss="huber", metrics=["mae"])
    return model

def get_evaluation(board, model):
    tensor = board_to_tensor(board)
    input_tensor = np.expand_dims(tensor, axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    return prediction[0][0]

def load_parquet(path: str, frac: int = 1) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.head(math.floor(df.shape[0] / frac))

class TensorGenerator:
    def __init__(self, df, batch_size=64):
        self.df = df
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.df))

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def get_batch(self, batch_idx):
        start = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        
        batch_df = self.df.iloc[start:end]
        
        X = np.zeros((self.batch_size, 8, 8, 19), dtype=np.float32)
        y = np.zeros((self.batch_size, 1), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            board = chess.Board(row["fen"])
            X[i] = board_to_tensor(board)
            y[i] = normalise_evaluation(row["cp"], row["mate"])
            
        return X, y

    def feed(self):
        """A generator function for model.fit()"""
        while True:
            np.random.shuffle(self.indexes)
            for i in range(self.__len__()):
                yield self.get_batch(i)


# --- Full Training Workfeed ---

fraction = 4
EPOCHS = 3

# Load Data
print("Loading data...")
df = load_parquet("data/train-00000-of-00017.parquet", fraction)
print("Data loaded:")
print(df.head())

# Split data into training and validation sets
print("Splitting data...")
train_df, val_df = train_test_split(df, test_size=0.1)
print("Data split")

# Setup Generators
print("Initialising generators")
train_gen = TensorGenerator(train_df, batch_size=128)
val_gen = TensorGenerator(val_df, batch_size=128)
print("Generators initialised")

# Build and Train
print("Building model...")
model = build_model()
model.summary()

print("Creating checkpoint...")
checkpoint_callback = ModelCheckpoint(
    filepath="models/tensorfish.keras",
    save_best_only=True,    # Only overwrite if the model is better than the previous version
    monitor="val_loss",     # Look at validation loss
    mode="min",             # Lower loss is better
    verbose=1
)

print("Fitting model...")
model.fit(
    train_gen.feed(),
    epochs=EPOCHS,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen.feed(),
    validation_steps=len(val_gen),
    callbacks=[checkpoint_callback]
)

model_path = "models/tensorfish.keras"
print(f"Model fit and saved to {model_path}")
model.save(model_path)

