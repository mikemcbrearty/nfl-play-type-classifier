import numpy as np
import random

def _to_one_hot(labels, dimension=4):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


def load_data():
    random.seed(1234)
    play_data_file = "play_data.csv"
    play_data = []
    with open(play_data_file, "r") as f:
        for line in f:
            row = line.strip().split(",")
            play_data.append(row)

    random.shuffle(play_data)
    play_data = np.array(play_data)
    len_play_data = play_data.shape[0]

    play_features = play_data[:, :-1].astype(float)
    play_targets = play_data[:, -1:].astype(int)
    play_targets.resize((len_play_data,))

    play_targets = _to_one_hot(play_targets)

    len_train_data = int(0.6 * len_play_data)
    len_val_data = int(0.2 * len_play_data)

    train_data = play_features[:len_train_data]
    val_data = play_features[len_train_data:len_train_data+len_val_data]
    test_data = play_features[len_train_data+len_val_data:]

    train_targets = play_targets[:len_train_data]
    val_targets = play_targets[len_train_data:len_train_data+len_val_data]
    test_targets = play_targets[len_train_data+len_val_data:]

    return (train_data, val_data, test_data), (train_targets, val_targets, test_targets)
